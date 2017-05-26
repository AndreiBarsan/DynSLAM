
#include <iostream>

#include <pangolin/pangolin.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <backward.hpp>
#include <GL/glut.h>

// TODO(andrei): Configure InfiniTAM's includes so we can be less verbose here.
#include "ImageSourceEngine.h"
#include "Utils.h"


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
using namespace InstRecLib::Reconstruction;
using namespace InstRecLib::Segmentation;

using namespace dynslam::utils;

static const int kUiWidth = 300;
static const float kPlotTimeIncrement = 0.1f;

// TODO(andrei): Move away from here.
/// \brief Main DynSLAM class interfacing between different submodules.
class DynSlam {

public:
    void Initialize(ITMMainEngine *itm_static_scene_engine_, ImageSourceEngine *image_source) {

      this->image_source_ = image_source;

      window_size_.x = image_source->getDepthImageSize().x;
      window_size_.y = image_source->getDepthImageSize().y;

      this->itm_static_scene_engine_ = itm_static_scene_engine_;
      this->current_frame_no_ = 0;

      bool allocate_gpu = true;
      Vector2i input_shape = image_source->getDepthImageSize();
      out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_rgb_image_= new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_raw_depth_image_ = new ITMShortImage(input_shape, true, allocate_gpu);

      // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.
      ITMSafeCall(cudaThreadSynchronize());

      cout << "DynSLAM initialization complete." << endl;
    }

  void ProcessFrame() {
    if (! image_source_->hasMoreImages()) {
      cout << "No more frames left in image source." << endl;
      return;
    }

    // Read the images from the first part of the pipeline
    image_source_->getImages(input_rgb_image_, input_raw_depth_image_);

    // Forward them to InfiniTAM for the background reconstruction.
    itm_static_scene_engine_->ProcessFrame(input_rgb_image_, input_raw_depth_image_);
    ITMSafeCall(cudaThreadSynchronize());

    current_frame_no_++;
  }

  const unsigned char* GetRaycastPreview() {
    // TODO(andrei): Get rid of reliance on itam enums via a driver abstraction.
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);
  }

  /// \brief Returns an **RGBA** preview of the latest color frame.
  const unsigned char* GetRgbPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_RGB);
  }

  /// \brief Returns an **RGBA** preview of the latest depth frame.
  const unsigned char* GetDepthPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_DEPTH);
  }

  /// \brief Returns an RGBA unsigned char frame containing the preview of the most recent frame's
  /// semantic segmentation.
  const unsigned char* GetSegmentationPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SEGMENTATION_RESULT);
  }

  /// \brief Returns an **RGBA** preview of the latest segmented object instance.
  const unsigned char* GetObjectPreview(int object_idx) {
//    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_INSTANCE_PREVIEW);
    ITMUChar4Image *preview = itm_static_scene_engine_->GetInstanceReconstructor()->GetInstancePreviewRGB(object_idx);
    if (nullptr == preview) {
      // This happens when there's no instances to preview.
      out_image_->Clear();
    } else {
      out_image_->SetFrom(preview, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
    }

    return out_image_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU)->getValues();
  }

  InstanceReconstructor * GetInstanceReconstructor() {
    return itm_static_scene_engine_->GetInstanceReconstructor();
  }

  int GetInputWidth() {
    return image_source_->getDepthImageSize().width;
  }

  int GetInputHeight() {
    return image_source_->getDepthImageSize().height;
  }

  int GetCurrentFrameNo() {
    return current_frame_no_;
  }

private:
  ITMLibSettings itm_lib_settings_;
  // TODO(andrei): Write custom image source.
  ImageSourceEngine *image_source_;

  // This is the main reconstruction component. Should split for dynamic+static.
  // In the future, we may need to write our own.
  // For now, this shall only handle reconstructing the static part of a scene.
  ITMMainEngine *itm_static_scene_engine_;

  ITMUChar4Image *out_image_;
  ITMUChar4Image *input_rgb_image_;
  ITMShortImage  *input_raw_depth_image_;

  int current_frame_no_;

  Vector2i window_size_;

  // TODO(andrei): Put this in a specific itam driver.
  const unsigned char* GetItamData(ITMMainEngine::GetImageType image_type) {
//    if (last_frame != current_frame_no_ ) {
//      last_frame = current_frame_no_;
    itm_static_scene_engine_->GetImage(out_image_, image_type);
//    }

    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }
};

/// TODO(andrei): Seriously consider using QT or wxWidgets. Pangolin is VERY limited in terms of the
/// widgets it supports. It doesn't even seem to support multiline text, or any reasonable way to
/// paint labels or lists or anything... Might be better not to worry too much about this, since
/// there isn't that much time...
class PangolinGui {
public:
  PangolinGui(DynSlam *dyn_slam) : dyn_slam_(dyn_slam) {
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
    delete detail_views;
    delete slam_preview;
    delete rgb_preview;
    delete depth_preview;
    delete segment_preview;
    delete object_preview;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);

      // TODO(andrei): Only use Pangolin camera if not PITA. Otherwise, can just base everything off
      // InfiniTAM's raycasting.
      main_view->Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      const unsigned char *slam_frame_data = dyn_slam_->GetRaycastPreview();

      // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance!
      slam_preview->Upload(slam_frame_data, GL_RGBA, GL_UNSIGNED_BYTE);
      slam_preview->RenderToViewport(true);

      rgb_view.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      rgb_preview->Upload(dyn_slam_->GetRgbPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      rgb_preview->RenderToViewport(true);

      depth_view.Activate();
      glColor3f(1.0, 0.0, 0.0);
      depth_preview->Upload(dyn_slam_->GetDepthPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      depth_preview->RenderToViewport(true);

      segment_view.Activate();
      glColor3f(1.0, 1.0, 1.0);
      segment_preview->Upload(dyn_slam_->GetSegmentationPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      segment_preview->RenderToViewport(true);
      DrawInstanceLables();

      // TODO(andrei): Make this an interactive point cloud/volume visualization.
      object_view.Activate();
      glColor3f(1.0, 1.0, 1.0);
      object_preview->Upload(dyn_slam_->GetObjectPreview(visualized_object_idx_),
                             GL_RGBA, GL_UNSIGNED_BYTE);
      object_preview->RenderToViewport(true);

      // dummy
//      graph_view.Activate();
//      glColor3f(1.0, 0.0, 0.0);
//      depth_preview->Upload(dyn_slam_->GetDepthPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
//      depth_preview->RenderToViewport(true);

      // dummy
      extra_view.Activate();
      glColor3f(1.0, 0.0, 0.0);
      depth_preview->Upload(dyn_slam_->GetDepthPreview(), GL_RGBA, GL_UNSIGNED_BYTE);
      depth_preview->RenderToViewport(true);

      plotter->Activate();

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

    const auto &instanceTracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
    for (const auto &track : instanceTracker.GetTracks()) {
      // Nothing to do for tracks with we didn't see this frame.
      if (track.GetLastFrame().frame_idx != dyn_slam_->GetCurrentFrameNo() - 1) {
        continue;
      }

      InstanceDetection latest_detection = track.GetLastFrame().instance_view.GetInstanceDetection();
      const auto &bbox = latest_detection.mask->GetBoundingBox();

      // Drawing the text requires converting from pixel coordinates to GL coordinates, which
      // range from (-1.0, -1.0) in the bottom-left, to (+1.0, +1.0) in the top-right.
      float panel_width = segment_view.GetBounds().w;
      float panel_height = segment_view.GetBounds().h;

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

    // GUI stuff
    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth));

    pangolin::Var<function<void(void)>> a_button("ui.Some Button", []() {
      cout << "Clicked the button!" << endl;
    });
    pangolin::Var<function<void(void)>> quit_button("ui.Quit Button", []() {
      pangolin::QuitAll();
    });
    reconstructions = new pangolin::Var<string>("ui.Rec", "");

    pangolin::Var<function<void(void)>> previous_object("ui.Previous Object", [&]() {
      cout << "Will select previous reconstructed object, once available..." << endl;
      SelectPreviousVisualizedObject();
    });
    pangolin::Var<function<void(void)>> next_object("ui.Next Object", [&]() {
      cout << "Will select next reconstructed object, once available..." << endl;
      SelectNextVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('n', [&]() {
      float val = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();
      active_instance_count_log_.Log(val);

      dyn_slam_->ProcessFrame();
    });

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(
      width, height, 420, 420, width / 2, height / 2, 0.1, 1000);

    float aspect_ratio = static_cast<float>(width) / height;
    rgb_view = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view = pangolin::Display("depth").SetAspect(aspect_ratio);
    segment_view = pangolin::Display("segment").SetAspect(aspect_ratio);
    object_view = pangolin::Display("object").SetAspect(aspect_ratio);
    extra_view = pangolin::Display("extra").SetAspect(aspect_ratio);

    // Storing pointers to these objects prevents a series of strange issues. The objects remain
    // under Pangolin's management, so they don't need to be deleted by the current class.
    main_view = &(pangolin::Display("main").SetAspect(aspect_ratio));
    detail_views = &(pangolin::Display("detail"));

    // Add labels to our data logs (and automatically to our plots).
    active_instance_count_log_.SetLabels({"Active tracks"});

    // OpenGL 'view' of data such as the number of actively tracked instances over time.
    plotter = new pangolin::Plotter(&active_instance_count_log_, 0.0f, 100.0f, -0.1f, 10.0f);

    // TODO(andrei): Maybe wrap these guys in another controller, make it an equal layout and
    // automagically support way more aspect ratios?
    main_view->SetBounds(pangolin::Attach::Pix(height * 1.5), 1.0,
                         pangolin::Attach::Pix(kUiWidth), 1.0);
    detail_views->SetBounds(0.0, pangolin::Attach::Pix(height * 1.5),
                            pangolin::Attach::Pix(kUiWidth), 1.0);
    detail_views->SetLayout(pangolin::LayoutEqual)
      .AddDisplay(rgb_view)
      .AddDisplay(depth_view)
      .AddDisplay(segment_view)
      .AddDisplay(object_view)
      .AddDisplay(*plotter)
      .AddDisplay(extra_view);

    // Internally, InfiniTAM stores these as RGBA, but we discard the alpha when we upload the
    // textures for visualization.
    this->slam_preview = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    this->rgb_preview = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    this->depth_preview = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    this->segment_preview = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    this->object_preview = new pangolin::GlTexture(width, height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
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
    int active_tracks = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();
    if (visualized_object_idx_ < active_tracks - 1) {
      visualized_object_idx_++;
    }
    else {
      visualized_object_idx_ = 0;
    }
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

private:
  DynSlam *dyn_slam_;

  /// Input frame dimensions. They dictate the overall window size.
  int width, height;

  pangolin::View *main_view;
  pangolin::View *detail_views;
  pangolin::View rgb_view;
  pangolin::View depth_view;
  pangolin::View segment_view;
  pangolin::View object_view;
//  pangolin::View graph_view;
  pangolin::View extra_view;  // TBD what we show here

  // Graph plotter and its data logger object
  pangolin::Plotter *plotter;
  pangolin::DataLog active_instance_count_log_;


  pangolin::GlTexture *dummy_image_texture;
  pangolin::GlTexture *slam_preview;
  pangolin::GlTexture *rgb_preview;
  pangolin::GlTexture *depth_preview;
  pangolin::GlTexture *segment_preview;
  pangolin::GlTexture *object_preview;

  pangolin::Var<string> *reconstructions;

  cv::Mat dummy_img;

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

}
}

int main(int argc, char **argv) {
  using namespace dynslam;

  gui::DynSlam *dyn_slam = new gui::DynSlam();
  const string dataset_root = "/home/andrei/work/libelas/cmake-build-debug/odo_seq_06/";
  const string calib_fpath = dataset_root + "calib.txt";
  const string rgb_image_format = dataset_root + "Frames/%04i.ppm";
  const string depth_image_format = dataset_root + "Frames/%04i.pgm";
  ImageSourceEngine *image_source = new ImageFileReader(
    calib_fpath.c_str(),
    rgb_image_format.c_str(),
    depth_image_format.c_str()
  );

  ITMLibSettings *settings = new ITMLibSettings();

  ITMMainEngine *static_scene_engine = new ITMMainEngine(
    settings,
    new ITMRGBDCalib(image_source->calib),
    image_source->getRGBImageSize(),
    image_source->getDepthImageSize()
  );

  dyn_slam->Initialize(static_scene_engine, image_source);

  gui::PangolinGui pango_gui(dyn_slam);
  pango_gui.Run();

  delete dyn_slam;
}

