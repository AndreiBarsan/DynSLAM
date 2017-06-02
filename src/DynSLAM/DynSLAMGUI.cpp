
#include <atomic>
#include <chrono>
#include <iostream>
#include <sys/time.h>

#include <backward.hpp>
#include <gflags/gflags.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <pangolin/pangolin.h>

#include "DynSlam.h"
#include "Utils.h"

#include "../pfmLib/ImageIOpfm.h"
#include "PrecomputedDepthEngine.h"

// Commandline arguments
DEFINE_string(dataset_root, "", "The root folder of the dataset to use.");

// TODO(andrei): Use [RIP] tags to signal spots where you wasted more than 30 minutes debugging a
// small, silly issue.

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
// TODO(andrei): this is a hack, please remove or depend on backward directly.
backward::SignalHandling sh;

/// \brief Define these because OpenCV doesn't. Used in the `cv::flip` OpenCV function.
enum {
  kCvFlipVertical = 0,
  kCvFlipHorizontal = 1,
  kCvFlipBoth = -1
};

namespace dynslam {
namespace gui {

using namespace std;
using namespace instreclib::reconstruction;
using namespace instreclib::segmentation;

using namespace dynslam;
using namespace dynslam::utils;

static const int kUiWidth = 300;
static const float kPlotTimeIncrement = 0.1f;

/// TODO(andrei): Seriously consider using QT or wxWidgets. Pangolin is VERY limited in terms of the
/// widgets it supports. It doesn't even seem to support multiline text, or any reasonable way to
/// paint labels or lists or anything... Might be better not to worry too much about this, since
/// there isn't that much time...
class PangolinGui {
public:
  PangolinGui(DynSlam *dyn_slam, Input *input)
      : dyn_slam_(dyn_slam),
        dyn_slam_input_(input)
  {
    cout << "Pangolin GUI initialized." << endl;

    // TODO(andrei): Proper scaling to save space and memory.
    width = dyn_slam->GetInputWidth(); // / 1.5;
    height = dyn_slam->GetInputHeight();// / 1.5;

    SetupDummyImage();
    CreatePangolinDisplays();
  }

  virtual ~PangolinGui() {
    delete reconstructions;
    delete dummy_image_texture;
    delete main_view;
    delete detail_views_;
    delete pane_texture;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);

      // TODO(andrei): You've ransacked this method quite a lot looking for that calibration bug on
      // June 1st. Please clean up!

      main_view->Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      const unsigned char *slam_frame_data = dyn_slam_->GetRaycastPreview();

      // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance, so you get dark, uncolored images.
//      pane_texture->Upload(slam_frame_data, GL_RGBA, GL_UNSIGNED_BYTE);
//      pane_texture->Upload(dyn_slam_->GetRgbPreview(), GL_RGBA, GL_UNSIGNED_BYTE);

      // Some experimental code for getting the camera to move on its own.
      if (wiggle_mode_->Get()) {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        double time_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        double time_scale = 1500.0;
        double r = 0.25;
        double cx = cos(time_ms / time_scale) * r;
        double cy = sin(time_ms / time_scale) * r - r * 2;
        double cz = sin(time_ms / time_scale) * r;
        pane_cam_->SetModelViewMatrix(
            pangolin::ModelViewLookAt(
                0, 0.0, 1.0,
                0, 0.5 + cy, -1.0,
                pangolin::AxisY)
        );
      }

      const unsigned char *preview = dyn_slam_->GetObjectRaycastFreeViewPreview(
          visualized_object_idx_,
          pane_cam_->GetModelViewMatrix());

      pane_texture->Upload(
          preview,
          GL_RGBA,
          GL_UNSIGNED_BYTE);
      pangolin::GlFont &font = pangolin::GlFont::I();
      pane_texture->RenderToViewport(true);
      font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-1.0, 0.9);

      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      pane_texture->Upload(dyn_slam_->GetRgbPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      pane_texture->RenderToViewport(true);

      depth_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      pane_texture->Upload(dyn_slam_->GetDepthPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      pane_texture->RenderToViewport(true);

      segment_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      if (nullptr != dyn_slam_->GetSegmentationPreview()) {
        pane_texture->Upload(dyn_slam_->GetSegmentationPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
        pane_texture->RenderToViewport(true);
        DrawInstanceLables();
      }

      object_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      pane_texture->Upload(dyn_slam_->GetObjectPreview(visualized_object_idx_),
                             GL_RGBA, GL_UNSIGNED_BYTE);
      pane_texture->RenderToViewport(true);

      object_reconstruction_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      pane_texture->Upload(
          dyn_slam_->GetObjectRaycastPreview(
              visualized_object_idx_,
              instance_cam_->GetModelViewMatrix()
          ),
          GL_RGBA,
          GL_UNSIGNED_BYTE);
      pane_texture->RenderToViewport(true);
      font.Text("Instance #%d", visualized_object_idx_).Draw(-1.0, 0.9);

      // Update various elements in the toolbar on the left.
      *(reconstructions) = Format(
        "%d active reconstructions",
        dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount()
      );

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

  /// \brief Renders informative labels regardin the currently active bounding boxes.
  /// Meant to be rendered over the segmentation preview window pane.
  void DrawInstanceLables() const {
    pangolin::GlFont &font = pangolin::GlFont::I();

    auto &instanceTracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
    for (const auto &pair: instanceTracker.GetActiveTracks()) {
      Track &track = instanceTracker.GetTrack(pair.first);
      // Nothing to do for tracks with we didn't see this frame.
      if (track.GetLastFrame().frame_idx != dyn_slam_->GetCurrentFrameNo() - 1) {
        continue;
      }

      InstanceDetection latest_detection = track.GetLastFrame().instance_view.GetInstanceDetection();
      const auto &bbox = latest_detection.mask->GetBoundingBox();

      // Drawing the text requires converting from pixel coordinates to GL coordinates, which
      // range from (-1.0, -1.0) in the bottom-left, to (+1.0, +1.0) in the top-right.
      float panel_width = segment_view_.GetBounds().w;
      float panel_height = segment_view_.GetBounds().h;

      float bbox_left = bbox.r.x0 - panel_width;
      float bbox_top = panel_height - bbox.r.y0 + font.Height();

      float gl_x = bbox_left / panel_width;
      float gl_y = bbox_top / panel_height;

      stringstream info_label;
      info_label << latest_detection.GetClassName() << "#" << track.GetId()
                 << "@" << setprecision(2)
                 << latest_detection.class_probability;
      glColor3f(1.0f, 0.0f, 0.0f);
      font.Text(info_label.str()).Draw(gl_x, gl_y, 0);
    }
  }

protected:
  /// \brief Creates the GUI layout and widgets.
  /// \note The layout is biased towards very wide images (~2:1 aspect ratio or more), which is very
  /// common in autonomous driving datasets.
  void CreatePangolinDisplays() {
    pangolin::CreateWindowAndBind("DynSLAM GUI",
                                  kUiWidth + width,
                                  // One full-height pane with the main preview, plus 3 * 0.5
                                  // height ones for various visualizations.
                                  static_cast<int>(ceil(height * 2.5)));

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // GUI components
    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth));

    pangolin::Var<function<void(void)>> a_button("ui.Save Static Map", [&]() {
      cout << "Saving static map..." << endl;
      dyn_slam_->SaveStaticMap();
      cout << "Done saving map." << endl;
    });
    reconstructions = new pangolin::Var<string>("ui.Rec", "");

    wiggle_mode_ = new pangolin::Var<bool>("ui.Wiggle mode", true);

    pangolin::Var<function<void(void)>> previous_object("ui.Previous Object", [&]() {
      cout << "Will select previous reconstructed object, once available..." << endl;
      SelectPreviousVisualizedObject();
    });
    pangolin::Var<function<void(void)>> next_object("ui.Next Object", [&]() {
      cout << "Will select next reconstructed object, once available..." << endl;
      SelectNextVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('n', [&]() {
      this->ProcessFrame();
    });
    pangolin::Var<function<void(void)>> quit_button("ui.Quit", []() {
      pangolin::QuitAll();
    });

    // This is used for the free view camera. The focal lengths are not used in rendering, BUT they
    // impact the sensitivity of the free view camera.
    proj_ = pangolin::ProjectionMatrix(width, height,
                                       1000, 1000,
                                       width / 2, height / 2,
                                       0.1, 1000);
    pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(0, 0, 0,
                                  0, 0, -1,
                                  pangolin::AxisY));

    // TODO(andrei): Set dynamically based on where instances are detected.
    instance_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(
          0.0, 0.0, 0.75,
          -1.5, 0.0, -1.0,
          pangolin::AxisY)
    );

    float aspect_ratio = static_cast<float>(width) / height;
    rgb_view_ = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view_ = pangolin::Display("depth").SetAspect(aspect_ratio);

    segment_view_ = pangolin::Display("segment").SetAspect(aspect_ratio);
    object_view_ = pangolin::Display("object").SetAspect(aspect_ratio);
    object_reconstruction_view_ = pangolin::Display("object_3d").SetAspect(aspect_ratio)
        .SetHandler(new pangolin::Handler3D(*instance_cam_));

    // Storing pointers to these objects prevents a series of strange issues. The objects remain
    // under Pangolin's management, so they don't need to be deleted by the current class.
    main_view = &(pangolin::Display("main").SetAspect(aspect_ratio));
    main_view->SetHandler(new pangolin::Handler3D(*pane_cam_));

    detail_views_ = &(pangolin::Display("detail"));

    // Add labels to our data logs (and automatically to our plots).
    data_log_.SetLabels({"Active tracks", "Free GPU Memory (100s of MiB)"});

    // OpenGL 'view' of data such as the number of actively tracked instances over time.
    float tick_x = 1.0f;
    float tick_y = 1.0f;
    plotter_ = new pangolin::Plotter(&data_log_, 0.0f, 100.0f, -0.1f, 25.0f, tick_x, tick_y);

    // TODO(andrei): Maybe wrap these guys in another controller, make it an equal layout and
    // automagically support way more aspect ratios?
    main_view->SetBounds(pangolin::Attach::Pix(height * 1.5), 1.0,
                         pangolin::Attach::Pix(kUiWidth), 1.0);
    detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height * 1.5),
                            pangolin::Attach::Pix(kUiWidth), 1.0);
    detail_views_->SetLayout(pangolin::LayoutEqual)
      .AddDisplay(rgb_view_)
      .AddDisplay(depth_view_)
      .AddDisplay(segment_view_)
      .AddDisplay(object_view_)
      .AddDisplay(*plotter_)
      .AddDisplay(object_reconstruction_view_);

    // Internally, InfiniTAM stores these as RGBA, but we discard the alpha when we upload the
    // textures for visualization (hence the 'GL_RGB' specification).
    this->pane_texture = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Yes, not a lot of
    // programs can load them natively for manual inspection. Photoshop can,
    cv::Mat im(370, 1226, CV_32FC1);
    cv::Mat preview(370, 1226, CV_8UC1);

    // FWIW it takes DispNet about 50 seconds to chew through a 1101 image kitti sequence at full
    // resolution (this includes the network initialization overhead AND dumping the images to the
    // disk). This means ~0.05s/frame, or 20 FPS.
//    const string pfm_path = "/home/barsana/datasets/kitti/odometry-dataset/sequences/06/precomputed-depth-dispnet/000000.pfm";
//    ReadFilePFM(im, pfm_path);
//
//    // Convert the 32-bit "HDR" depth into something we can display.
//    im.convertTo(preview, CV_8UC1, 1.0);
//
//    cv::imshow("PFM depth map preview", preview);
//    cv::waitKey(0);

    cout << "Pangolin UI setup complete." << endl;
  }

  void SetupDummyImage() {
    cout << "Loading George..." << endl;
    dummy_img = cv::imread("/home/andrei/Pictures/george.jpg");

    cv::flip(dummy_img, dummy_img, kCvFlipVertical);
    const int george_width = dummy_img.cols;
    const int george_height = dummy_img.rows;
    this->dummy_image_texture = new pangolin::GlTexture(
      george_width, george_height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  }

  void SelectNextVisualizedObject() {
    // We pick the closest next object (by ID). We need to do this because some tracks may no
    // longer be available.
    InstanceTracker &tracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
    int closest_next_id = -1;
    int closest_next_delta = std::numeric_limits<int>::max();

    if (tracker.GetActiveTrackCount() == 0) {
      visualized_object_idx_ = 0;
      return;
    }

    for(const auto &pair : tracker.GetActiveTracks()) {
      int id = pair.first;
      int delta = id - visualized_object_idx_;
      if (delta < closest_next_delta && delta != 0) {
        closest_next_delta = delta;
        closest_next_id = id;
      }
    }

    visualized_object_idx_ = closest_next_id;
  }

  void SelectPreviousVisualizedObject() {
    int active_tracks = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();
    if (visualized_object_idx_ <= 0) {
      visualized_object_idx_ = active_tracks - 1;
    }
    else {
      visualized_object_idx_--;
    }
  }

  /// \brief Advances to the next input frame, and integrates it into the map.
  void ProcessFrame() {
    active_object_count_ = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();

    size_t free_gpu_memory_bytes;
    size_t total_gpu_memory_bytes;
    cudaMemGetInfo(&free_gpu_memory_bytes, &total_gpu_memory_bytes);

    const double kBytesToGb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    double free_gpu_gb = static_cast<float>(free_gpu_memory_bytes) * kBytesToGb;
    data_log_.Log(
        active_object_count_,
        static_cast<float>(free_gpu_gb) * 10.0f    // Mini-hack to make the scales better
    );

    // Main workhorse function of the underlying SLAM system.
    dyn_slam_->ProcessFrame(this->dyn_slam_input_);
  }

private:
  DynSlam *dyn_slam_;
  Input *dyn_slam_input_;

  /// Input frame dimensions. They dictate the overall window size.
  int width, height;

  pangolin::View *main_view;
  pangolin::View *detail_views_;
  pangolin::View rgb_view_;
  pangolin::View depth_view_;
  pangolin::View segment_view_;
  pangolin::View object_view_;
  pangolin::View object_reconstruction_view_;

  pangolin::OpenGlMatrix proj_;
  pangolin::OpenGlRenderState *pane_cam_;
  pangolin::OpenGlRenderState *instance_cam_;

  // Graph plotter and its data logger object
  pangolin::Plotter *plotter_;
  pangolin::DataLog data_log_;

  pangolin::GlTexture *dummy_image_texture;
  pangolin::GlTexture *pane_texture;

  pangolin::Var<string> *reconstructions;

  cv::Mat dummy_img;

  // Atomic because it gets set from a UI callback. Technically, Pangolin shouldn't invoke callbacks
  // from a different thread, but using atomics for this is generally a good practice anyway.
  atomic<int> active_object_count_;

  /// \brief Whether the 3D scene view should be automatically moving around.
  /// If this is off, then the user has control over the camera.
  pangolin::Var<bool> *wiggle_mode_;

  // Indicates which object is currently being visualized in the GUI.
  int visualized_object_idx_ = 0;

  void UploadDummyTexture() {
    // Mess with George's bytes a little bit for OpenGL <-> OpenCV compatibility.
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (dummy_img.step & 3) ? 1 : 4);
    //set length of one complete row in data (doesn't need to equal img.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img.step/dummy_img.elemSize());
    dummy_image_texture->Upload(dummy_img.data, GL_BGR, GL_UNSIGNED_BYTE);
  }
};

} // namespace gui

/// \brief Constructs a DynSLAM instance to run on a KITTI Odometry dataset sequence, using the
///        ground truth pose information from the Inertial Navigation System (INS) instead of any
///        visual odometry.
/// This is useful when you want to focus on the quality of the reconstruction, instead of that of
/// the odometry.
void BuildDynSlamKittiOdometryGT(const string &dataset_root, DynSlam **dyn_slam_out, Input **input_out) {
  auto calib = ReadITMCalibration(dataset_root + "/itm-calib.txt");
  *input_out = new Input(
      dataset_root,
      new PrecomputedDepthEngine(dataset_root + "/precomputed-depth/Frames/", "%04d.pgm"),
      calib);

  *dyn_slam_out = new gui::DynSlam();
  cv::Vec2i rgb_size(static_cast<int>(calib.intrinsics_rgb.sizeX),
                     static_cast<int>(calib.intrinsics_rgb.sizeY));
  cv::Vec2i depth_size(static_cast<int>(calib.intrinsics_d.sizeX),
                       static_cast<int>(calib.intrinsics_d.sizeY));

  // [RIP] I lost a couple of hours debugging a bug caused by the fact that InfiniTAM still works
  // even when there is a discrepancy between the size of the depth/rgb inputs, as specified in the
  // calibration file, and the actual size of the input images.

  ITMLibSettings *settings = new ITMLibSettings();
  // TODO(andrei): Add this to dataset prep script!
  settings->groundTruthPoseFpath = dataset_root + "/poses.txt";

  drivers::InfiniTamDriver *driver = new InfiniTamDriver(
      settings,
      new ITMRGBDCalib(calib),
      ToItmVec(rgb_size),
      ToItmVec(depth_size));

  const string seg_folder = dataset_root + "/seg_image_2/mnc";
  auto segmentation_provider =
      new instreclib::segmentation::PrecomputedSegmentationProvider(seg_folder);

  (*dyn_slam_out)->Initialize(driver, segmentation_provider);
}

} // namespace dynslam

int main(int argc, char **argv) {
  using namespace dynslam;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const string dataset_root = FLAGS_dataset_root;
  if (dataset_root.empty()) {
    cerr << "Please specify a dataset to work with. The --dataset_root=<path> flag must be set."
         << endl;

    return -1;
  }

  DynSlam *dyn_slam;
  Input *input;
  BuildDynSlamKittiOdometryGT(dataset_root, &dyn_slam, &input);

  gui::PangolinGui pango_gui(dyn_slam, input);
  pango_gui.Run();

  delete dyn_slam;
  delete input;
}

