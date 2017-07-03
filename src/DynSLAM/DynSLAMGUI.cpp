
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

//#include <backward.hpp>
#include <gflags/gflags.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <pangolin/pangolin.h>

#include "DynSlam.h"
#include "Utils.h"

#include "../pfmLib/ImageIOpfm.h"
#include "PrecomputedDepthProvider.h"
#include "InstRecLib/VisoSparseSFProvider.h"
#include "DSHandler3D.h"


const std::string kKittiOdometry = "kitti-odometry";
const std::string kKitti         = "kitti";
const std::string kCityscapes    = "cityscapes";

// Commandline arguments using gflags
DEFINE_string(dataset_type,
              kKittiOdometry,
              "The type of the input dataset at which 'dataset_root'is pointing. "
              "Currently supported is only 'kitti-odometry'.");
DEFINE_string(dataset_root, "", "The root folder of the dataset sequence to use.");

// Useful offsets for dynamic object reconstruction:
//  int frame_offset = 85; // for odo seq 02
//  int frame_offset = 4015;         // Clear dynamic object in odometry sequence 08.
DEFINE_int32 (frame_offset, 0, "The frame index from which to start reading the dataset sequence.");

// Note: the [RIP] tags signal spots where I wasted more than 30 minutes debugging a small, silly
// issue.

// TODO(andrei): Consider making the libviso module implement two interfaces
// with the same underlying engine: visual odometry and sparse scene flow. You
// could then feed this sparse flow (prolly wrapped inside an Eigen matrix or
// something) into the instance reconstruction, which would then associate it
// with instances, and run the more refined tracker, etc. In the first stage we
// should get the coarse dynamic object pose estimation going, and then add the
// refinement.

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
// TODO(andrei): this is a hack, please remove or depend on backward directly.
//backward::SignalHandling sh;

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

/// \brief The main GUI and entry point for DynSLAM.
class PangolinGui {
public:
  PangolinGui(DynSlam *dyn_slam, Input *input)
      : dyn_slam_(dyn_slam),
        dyn_slam_input_(input)
  {
    cout << "Pangolin GUI initializing..." << endl;

    // TODO(andrei): Proper scaling to save space and memory.
    width_ = dyn_slam->GetInputWidth(); // / 1.5;
    height_ = dyn_slam->GetInputHeight();// / 1.5;

    SetupDummyImage();
    CreatePangolinDisplays();

    cout << "Pangolin GUI initialized." << endl;
  }

  virtual ~PangolinGui() {
    // No need to delete any view pointers; Pangolin deletes those itself on shutdown.
    delete dummy_image_texture_;
    delete pane_texture_;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);

      if (autoplay_->Get()) {
        ProcessFrame();
      }

      main_view_->Activate();
      glColor3f(1.0f, 1.0f, 1.0f);

      if(live_raycast_->Get()) {
        const unsigned char *slam_frame_data = dyn_slam_->GetRaycastPreview();
        pane_texture_->Upload(slam_frame_data, GL_RGBA, GL_UNSIGNED_BYTE);
      }
      else {
        // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance,
        // so you get dark, uncolored images.

        // Some experimental code for getting the camera to move on its own.
        if (wiggle_mode_->Get()) {
          struct timeval tp;
          gettimeofday(&tp, NULL);
          double time_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
          double time_scale = 1500.0;
          double r = 0.2;
//          double cx = cos(time_ms / time_scale) * r;
          double cy = sin(time_ms / time_scale) * r - r * 2;
//          double cz = sin(time_ms / time_scale) * r;
          pane_cam_->SetModelViewMatrix(
              pangolin::ModelViewLookAt(
                  0, 0.0, 0.2,
                  0, 0.5 + cy, -1.0,
                  pangolin::AxisY)
          );
        }

//        struct timeval tp;
//        gettimeofday(&tp, NULL);
//        double time_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
//        double time_scale = 500.0;
//        double r = 1;
//        double a = sin(time_ms / time_scale) * r;
//        double cc = cos(time_ms / time_scale) * r;
//        cout << (sin(time_ms/time_scale) * 10) << endl;
//        instance_cam_->SetModelViewMatrix(
//        pangolin::ModelViewLookAt(
//            -3.0 + a, -1.00,  12.75,
//            -3.0, -1.10,  10.00,
//            pangolin::AxisY)
//        );

        if (dyn_slam_->GetCurrentFrameNo() > 1) {
          const unsigned char *preview = dyn_slam_->GetStaticMapRaycastPreview(
              visualized_object_idx_,
              pane_cam_->GetModelViewMatrix(),
              static_cast<PreviewType>(current_preview_type_));
          pane_texture_->Upload(preview, GL_RGBA, GL_UNSIGNED_BYTE);
        }
      }
      pangolin::GlFont &font = pangolin::GlFont::I();
      pane_texture_->RenderToViewport(true);
      font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.95, 0.9);

      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      if (display_raw_previews_->Get()) {
        UploadCvTexture(*(dyn_slam_->GetRgbPreview()), *pane_texture_);
      }
      else {
        UploadCvTexture(*(dyn_slam_->GetStaticRgbPreview()), *pane_texture_);
      }
      pane_texture_->RenderToViewport(true);

      if (dyn_slam_->GetCurrentFrameNo() > 1 && preview_sf_->Get()) {
        PreviewSparseSF(dyn_slam_->GetLatestFlow().matches, rgb_view_);
      }

      depth_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      if (display_raw_previews_->Get()) {
        UploadCvTexture(*(dyn_slam_->GetDepthPreview()), *pane_texture_, false, GL_SHORT);
      }
      else {
        UploadCvTexture(*(dyn_slam_->GetStaticDepthPreview()), *pane_texture_, false, GL_SHORT);
      }
      pane_texture_->RenderToViewport(true);

      segment_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      if (nullptr != dyn_slam_->GetSegmentationPreview()) {
        UploadCvTexture(*dyn_slam_->GetSegmentationPreview(), *pane_texture_);
        pane_texture_->RenderToViewport(true);
        DrawInstanceLables();
      }

      object_view_.Activate();
      glColor4f(1.0, 1.0, 1.0, 1.0f);
      pane_texture_->Upload(dyn_slam_->GetObjectPreview(visualized_object_idx_),
                             GL_RGBA, GL_UNSIGNED_BYTE);
      pane_texture_->RenderToViewport(true);

      auto &tracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
      if (dyn_slam_->GetCurrentFrameNo() > 0 && preview_sf_->Get()) {
        // TODO-LOW(andrei): This is bonkers. Add some helpers!
        if (tracker.HasTrack(visualized_object_idx_)) {
          const auto &track = tracker.GetTrack(visualized_object_idx_);
          const auto &instance_flow = track.GetLastFrame().instance_view.GetFlow();
          PreviewSparseSF(instance_flow, object_view_);
        }
      }

      object_reconstruction_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      pane_texture_->Upload(
          dyn_slam_->GetObjectRaycastPreview(
              visualized_object_idx_,
              instance_cam_->GetModelViewMatrix(),
              static_cast<PreviewType>(current_preview_type_)
          ),
          GL_RGBA,
          GL_UNSIGNED_BYTE);
      pane_texture_->RenderToViewport(true);
      font.Text("Instance #%d", visualized_object_idx_).Draw(-1.0, 0.9);

      // Update various elements in the toolbar on the left.
      *(reconstructions) = Format(
        "%d active reconstructions",
        dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount()
      );

      // Disable autoplay once we reach the end of a sequence.
      if (! this->dyn_slam_input_->HasMoreImages()) {
        (*this->autoplay_) = false;
      }

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

  /// \brief Converts the pixel coordinates into [-1, +1]-style OpenGL coordinates.
  /// \note The GL coordinates range from (-1.0, -1.0) in the bottom-left, to (+1.0, +1.0) in the
  ///       top-right.
  cv::Vec2f PixelsToGl(Eigen::Vector2f px, Eigen::Vector2f px_range, const pangolin::View &view) {
    float view_w = view.GetBounds().w;
    float view_h = view.GetBounds().h;

    float px_x = px(0) * (view_w / px_range(0));
    float px_y = px(1) * (view_h / px_range(1));

    float gl_x = ((px_x - view_w) / view_w + 0.5f) * 2.0f;
    // We need the opposite sign here since pixel coordinates are assumed to have the origin in the
    // top-left, while the GL origin is in the bottom-left.
    float gl_y = ((view_h - px_y) / view_h - 0.5f) * 2.0f;

    return cv::Vec2f(gl_x, gl_y);
  }

  /// \brief Renders informative labels regardin the currently active bounding boxes.
  /// Meant to be rendered over the segmentation preview window pane.
  void DrawInstanceLables() {
    pangolin::GlFont &font = pangolin::GlFont::I();

    auto &instanceTracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
    for (const auto &pair: instanceTracker.GetActiveTracks()) {
      Track &track = instanceTracker.GetTrack(pair.first);
      // Nothing to do for tracks with we didn't see this frame.
      if (track.GetLastFrame().frame_idx != dyn_slam_->GetCurrentFrameNo() - 2) { // TODO(andrei): Why this index gap of 2?
        continue;
      }

      InstanceDetection latest_detection = track.GetLastFrame().instance_view.GetInstanceDetection();
      const auto &bbox = latest_detection.mask->GetBoundingBox();
      cv::Vec2f gl_pos = PixelsToGl(Eigen::Vector2f(bbox.r.x0, bbox.r.y0 - font.Height()),
                                    Eigen::Vector2f(width_, height_),
                                    segment_view_);

      stringstream info_label;
      info_label << latest_detection.GetClassName() << "#" << track.GetId()
                 << "@" << setprecision(2)
                 << latest_detection.class_probability;
      glColor3f(1.0f, 0.0f, 0.0f);
      font.Text(info_label.str()).Draw(gl_pos[0], gl_pos[1], 0);
    }
  }

  /// \brief Renders a simple preview of the scene flow information onto the currently active pane.
  void PreviewSparseSF(const vector<RawFlow> &flow, const pangolin::View &view) {
    pangolin::GlFont &font = pangolin::GlFont::I();
    Eigen::Vector2f frame_size(width_, height_);
    font.Text("libviso2 scene flow preview").Draw(-0.98f, 0.89f);

    // We don't need z-checks since we're rendering UI stuff.
    glDisable(GL_DEPTH_TEST);
    for(const RawFlow &match : flow) {
      cv::Vec2f gl_pos = PixelsToGl(match.curr_left, frame_size, view);
      cv::Vec2f gl_pos_old = PixelsToGl(match.prev_left, frame_size, view);

      cv::Vec2f delta = gl_pos - gl_pos_old;
      float magnitude = 15.0f * static_cast<float>(cv::norm(delta));

      glColor4f(0.3f, 0.3f, 0.9f, 1.0f);
      pangolin::glDrawCross(gl_pos[0], gl_pos[1], 0.025f);
      glColor4f(max(0.2f, min(1.0f, magnitude)), 0.4f, 0.4f, 1.0f);
      pangolin::glDrawLine(gl_pos_old[0], gl_pos_old[1], gl_pos[0], gl_pos[1]);
    }

    glEnable(GL_DEPTH_TEST);
  }

protected:
  /// \brief Creates the GUI layout and widgets.
  /// \note The layout is biased towards very wide images (~2:1 aspect ratio or more), which is very
  /// common in autonomous driving datasets.
  void CreatePangolinDisplays() {
    pangolin::CreateWindowAndBind("DynSLAM GUI",
                                  kUiWidth + width_,
                                  // One full-height pane with the main preview, plus 3 * 0.5
                                  // height ones for various visualizations.
                                  static_cast<int>(ceil(height_ * 2.5)));

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // GUI components
    pangolin::CreatePanel("ui")
      .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth));

    auto next_frame = [this]() {
      *(this->autoplay_) = false;
      this->ProcessFrame();
    };
    pangolin::Var<function<void(void)>> next_frame_button("ui.[N]ext Frame", next_frame);
    pangolin::RegisterKeyPressCallback('n', next_frame);

    auto save_map = [this]() {
      Tic("Static map mesh generation");
      cout << "Saving static map..." << endl;
      dyn_slam_->SaveStaticMap(dyn_slam_input_->GetName(),
                               dyn_slam_input_->GetDepthProvider()->GetName());
      cout << "Done saving map." << endl;
      Toc();
    };

    pangolin::Var<function<void(void)>> save_map_button("ui.[S]ave Static Map", save_map);
    pangolin::RegisterKeyPressCallback('s', save_map);

    reconstructions = new pangolin::Var<string>("ui.Rec", "");

    wiggle_mode_ = new pangolin::Var<bool>("ui.Wiggle mode", false, true);
    autoplay_ = new pangolin::Var<bool>("ui.[A]utoplay", false, true);
    pangolin::RegisterKeyPressCallback('a', [this]() {
      *(this->autoplay_) = ! *(this->autoplay_);
    });
    live_raycast_ = new pangolin::Var<bool>("ui.Raycast Mode", false, true);
    display_raw_previews_ = new pangolin::Var<bool>("ui.Raw Previews", true, true);
    preview_sf_ = new pangolin::Var<bool>("ui.Show Scene Flow", false, true);

    pangolin::Var<function<void(void)>> previous_object("ui.Previous Object [z]", [this]() {
      SelectPreviousVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('z', [this]() { SelectPreviousVisualizedObject(); });
    pangolin::Var<function<void(void)>> next_object("ui.Ne[x]t Object", [this]() {
      SelectNextVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('x', [this]() { SelectNextVisualizedObject(); });
    auto save_object = [this]() {
      dyn_slam_->SaveDynamicObject(dyn_slam_input_->GetName(),
                                     dyn_slam_input_->GetDepthProvider()->GetName(),
                                     visualized_object_idx_);
    };
    pangolin::Var<function<void(void)>> save_active_object("ui.Save Active [O]bject", save_object);
    pangolin::RegisterKeyPressCallback('o', save_object);

    auto quit = [this]() { pangolin::QuitAll(); };
    pangolin::Var<function<void(void)>> quit_button("ui.[Q]uit", quit);
    pangolin::RegisterKeyPressCallback('q', quit);

    auto previous_preview_type = [this]() {
      if (--current_preview_type_ < 0) {
        current_preview_type_ = (PreviewType::kEnd - 1);
      }
    };
    auto next_preview_type = [this]() {
      if (++current_preview_type_ >= PreviewType::kEnd) {
        current_preview_type_ = 0;
      }
    };
    pangolin::Var<function<void(void)>> ppt("ui.Previous Preview Type [j]", previous_preview_type);
    pangolin::RegisterKeyPressCallback('j', previous_preview_type);
    pangolin::Var<function<void(void)>> npt("ui.Next Preview Type [k]", next_preview_type);
    pangolin::RegisterKeyPressCallback('k', next_preview_type);

    // This is used for the free view camera. The focal lengths are not used in rendering, BUT they
    // impact the sensitivity of the free view camera. The smaller they are, the faster the camera
    // responds to input (ideally, you should use the translation and zoom scales to control this,
    // though).
    float cam_focal_length = 250.0f;
    proj_ = pangolin::ProjectionMatrix(width_, height_,
                                       cam_focal_length, cam_focal_length,
                                       width_ / 2, height_ / 2,
                                       0.1, 1000);

    // TODO(andrei): Set based on initial camera pose, to account for starting the reconstruction
    // from a mid-frame, instead of the first one.
    pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(0, 0, 0,
                                  0, 0, -1,
        // -Y is up.
//        pangolin::ModelViewLookAt(0, -7.5, 290,
//                                  0, -7.5, 289,
                                  pangolin::AxisY));

    instance_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(
          -3.0, -1.00,  12.75,
          -3.0, -1.00,  10.0,
          pangolin::AxisY)
    );

    float aspect_ratio = static_cast<float>(width_) / height_;
    rgb_view_ = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view_ = pangolin::Display("depth").SetAspect(aspect_ratio);

    segment_view_ = pangolin::Display("segment").SetAspect(aspect_ratio);
    object_view_ = pangolin::Display("object").SetAspect(aspect_ratio);
    float camera_translation_scale = 0.01f;
    float camera_zoom_scale = 0.01f;

    object_reconstruction_view_ = pangolin::Display("object_3d").SetAspect(aspect_ratio)
        .SetHandler(new DSHandler3D(
            *instance_cam_,
            pangolin::AxisY,
            camera_translation_scale,
            camera_zoom_scale
        ));

    // These objects remain under Pangolin's management, so they don't need to be deleted by the
    // current class.
    main_view_ = &(pangolin::Display("main").SetAspect(aspect_ratio));
    main_view_->SetHandler(new DSHandler3D(*pane_cam_,
                                           pangolin::AxisY,
                                           camera_translation_scale,
                                           camera_zoom_scale));

    detail_views_ = &(pangolin::Display("detail"));

    // Add labels to our data logs (and automatically to our plots).
    data_log_.SetLabels({"Active tracks",
                         "Free GPU Memory (100s of MiB)",
                         "Static map memory usage (100s of MiB)",
                         "Static map memory usage without decay (100s of Mib)",
                        });

    // OpenGL 'view' of data such as the number of actively tracked instances over time.
    float tick_x = 1.0f;
    float tick_y = 1.0f;
    plotter_ = new pangolin::Plotter(&data_log_, 0.0f, 200.0f, -0.1f, 25.0f, tick_x, tick_y);
    plotter_->Track("$i");  // This enables automatic scrolling for the live plots.

    // TODO(andrei): Maybe wrap these guys in another controller, make it an equal layout and
    // automagically support way more aspect ratios?
    main_view_->SetBounds(pangolin::Attach::Pix(height_ * 1.5), 1.0,
                         pangolin::Attach::Pix(kUiWidth), 1.0);
    detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height_ * 1.5),
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
    this->pane_texture_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RGB,
                                                  GL_UNSIGNED_BYTE);
    cout << "Pangolin UI setup complete." << endl;
  }

  void SetupDummyImage() {
    dummy_img_ = cv::imread("../data/george.jpg");
    const int george_width = dummy_img_.cols;
    const int george_height = dummy_img_.rows;
    this->dummy_image_texture_ = new pangolin::GlTexture(
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
    cout << endl << "[Starting frame " << dyn_slam_->GetCurrentFrameNo() + 1 << "]" << endl;
    active_object_count_ = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();

    size_t free_gpu_memory_bytes;
    size_t total_gpu_memory_bytes;
    cudaMemGetInfo(&free_gpu_memory_bytes, &total_gpu_memory_bytes);

    const double kBytesToGb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    double free_gpu_gb = static_cast<float>(free_gpu_memory_bytes) * kBytesToGb;
    data_log_.Log(
        active_object_count_,
        static_cast<float>(free_gpu_gb) * 10.0f,   // Mini-hack to make the scales better
        dyn_slam_->GetStaticMapMemory() * 10.0f * kBytesToGb,
        (dyn_slam_->GetStaticMapMemory() + dyn_slam_->GetStaticMapSavedDecayMemory()) * 10.0f * kBytesToGb
    );

    Tic("DynSLAM frame");
    // Main workhorse function of the underlying SLAM system.
    dyn_slam_->ProcessFrame(this->dyn_slam_input_);
    int64_t frame_time_ms = Toc(true);
    float fps = 1000.0f / static_cast<float>(frame_time_ms);
    cout << "[Finished frame " << dyn_slam_->GetCurrentFrameNo() << " in " << frame_time_ms
         << "ms @ " << setprecision(4) << fps << " FPS (approx.)]"
         << endl;
  }

private:
  DynSlam *dyn_slam_;
  Input *dyn_slam_input_;

  /// Input frame dimensions. They dictate the overall window size.
  int width_, height_;

  pangolin::View *main_view_;
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

  pangolin::GlTexture *dummy_image_texture_;
  pangolin::GlTexture *pane_texture_;

  pangolin::Var<string> *reconstructions;

  /// \brief Used for UI testing whenever necessary. Not related to the SLAM system in any way.
  cv::Mat dummy_img_;

  // Atomic because it gets set from a UI callback. Technically, Pangolin shouldn't invoke callbacks
  // from a different thread, but using atomics for this is generally a good practice anyway.
  atomic<int> active_object_count_;

  /// \brief Whether the 3D scene view should be automatically moving around.
  /// If this is off, then the user has control over the camera.
  pangolin::Var<bool> *wiggle_mode_;
  /// \brief When this is on, the input gets processed as fast as possible, without requiring any
  /// user input.
  pangolin::Var<bool> *autoplay_;
  /// \brief If enabled, the latest frame's raycast is displayed in the main pane. Otherwise, a
  /// free-roam view of the scene is provided.
  pangolin::Var<bool> *live_raycast_;
  /// \brief Whether to display the RGB and depth previews directly from the input, or from the
  /// static scene, i.e., with the dynamic objects removed.
  pangolin::Var<bool> *display_raw_previews_;
  /// \brief Whether to preview the sparse scene flow on the input and current instance RGP panes.
  pangolin::Var<bool> *preview_sf_;

  // TODO(andrei): On-the-fly depth provider toggling.
  // TODO(andrei): Reset button.
  // TODO(andrei): Dynamically set depth range.

  // Indicates which object is currently being visualized in the GUI.
  int visualized_object_idx_ = 0;

  int current_preview_type_ = kColor;

  /// \brief Prepares the contents of an OpenCV Mat object for rendering with Pangolin (OpenGL).
  /// Does not actually render the texture.
  static void UploadCvTexture(
      const cv::Mat &mat,
      pangolin::GlTexture &texture,
      bool color = true,
      GLenum data_type = GL_UNSIGNED_BYTE
  ) {
    int old_alignment, old_row_length;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_alignment);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &old_row_length);

    int new_alignment = (mat.step & 3) ? 1 : 4;
    int new_row_length = static_cast<int>(mat.step / mat.elemSize());
    glPixelStorei(GL_UNPACK_ALIGNMENT, new_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, new_row_length);

    GLenum data_format = (color) ? GL_BGR : GL_GREEN;
    texture.Upload(mat.data, data_format, data_type);

    glPixelStorei(GL_UNPACK_ALIGNMENT, old_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, old_row_length);
  }

  void UploadDummyTexture() {
    // Mess with the bytes a little bit for OpenGL <-> OpenCV compatibility.
    // use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (dummy_img_.step & 3) ? 1 : 4);
    //set length of one complete row in data (doesn't need to equal img.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img_.step/dummy_img_.elemSize());
    dummy_image_texture_->Upload(dummy_img_.data, GL_BGR, GL_UNSIGNED_BYTE);
  }
};

} // namespace gui

/// \brief Constructs a DynSLAM instance to run on a KITTI Odometry dataset sequence, using the
///        ground truth pose information from the Inertial Navigation System (INS) instead of any
///        visual odometry.
/// This is useful when you want to focus on the quality of the reconstruction, instead of that of
/// the odometry.
void BuildDynSlamKittiOdometryGT(const string &dataset_root, DynSlam **dyn_slam_out, Input **input_out) {
  // Parameters used in the KITTI-odometry dataset.
  // TODO(andrei): Read from a file.
  float baseline_m = 0.537150654273f;
  float focal_length_px = 707.0912f;
  StereoCalibration stereo_calibration(baseline_m, focal_length_px);

//  Input::Config input_config = Input::KittiOdometryConfig();
  Input::Config input_config = Input::KittiOdometryDispnetConfig();
  auto itm_calibration = ReadITMCalibration(dataset_root + "/" + input_config.itm_calibration_fname);

  int frame_offset = FLAGS_frame_offset;

  *input_out = new Input(
      dataset_root,
      input_config,
      // TODO(andrei): Why not both elas and dispnet? Maybe it's worth investigating to use the more
      // conservative, sharper libelas depth maps for the main map, but the smoother ones produced
      // by dispnet for the objects. This may be a good idea since libelas tends to screw up when
      // faced with reflective surfaces, but dispnet is more robust to that. Similarly, for certain
      // areas of the static map such as foliage and fences, dispnet's smoothness is more of a
      // liability than an asset.
      // TODO(andrei): Carefully read the dispnet paper.
      new PrecomputedDepthProvider(dataset_root + "/" + input_config.depth_folder,
                                   input_config.depth_fname_format,
                                   input_config.read_depth,
                                   frame_offset),
      itm_calibration,
      stereo_calibration,
      frame_offset);

  // [RIP] I lost a couple of hours debugging a bug caused by the fact that InfiniTAM still works
  // even when there is a discrepancy between the size of the depth/rgb inputs, as specified in the
  // calibration file, and the actual size of the input images (but it screws up the previews).

  ITMLibSettings *settings = new ITMLibSettings();
  settings->groundTruthPoseFpath = dataset_root + "/" + input_config.odometry_fname;
  settings->groundTruthPoseOffset = frame_offset;

  drivers::InfiniTamDriver *driver = new InfiniTamDriver(
      settings,
      new ITMRGBDCalib(itm_calibration),
      ToItmVec((*input_out)->GetRgbSize()),
      ToItmVec((*input_out)->GetDepthSize()));

  const string seg_folder = dataset_root + "/" + input_config.segmentation_folder;
  auto segmentation_provider =
      new instreclib::segmentation::PrecomputedSegmentationProvider(seg_folder, frame_offset);

  VisualOdometryStereo::parameters sf_params;
  // TODO(andrei): The main VO (which we're not using viso2 for, at the moment (June '17) and the
  // "VO" we use to align object instance frames have VASTLY different requirements, so we should
  // use separate parameter sets for them.
  sf_params.base = baseline_m;
  sf_params.match.nms_n = 3;    // Optimal from KITTI leaderboard: 3 (also the default)
  sf_params.match.half_resolution = 0;
  sf_params.match.multi_stage = 1;    // Default = 1 (= 0 => much slower)
  sf_params.match.refinement = 1;   // Default = 1 (per-pixel); 2 = sub-pixel, slower
  sf_params.ransac_iters = 500;    // Default = 200; added more to see if it helps instance reconstruction
  sf_params.inlier_threshold = 5.0;   // Default = 2.0 => we attempt to be coarser for the sake of reconstructing
                                      // object instances
  sf_params.bucket.max_features = 10;    // Default = 2
  sf_params.calib.cu = itm_calibration.intrinsics_rgb.projectionParamsSimple.px;
  sf_params.calib.cv = itm_calibration.intrinsics_rgb.projectionParamsSimple.py;
  sf_params.calib.f  = itm_calibration.intrinsics_rgb.projectionParamsSimple.fx; // TODO should we average fx and fy?

  auto sparse_sf_provider = new instreclib::VisoSparseSFProvider(sf_params);

  *dyn_slam_out = new gui::DynSlam();
  (*dyn_slam_out)->Initialize(driver, segmentation_provider, sparse_sf_provider);
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

  if (FLAGS_dataset_type != kKittiOdometry) {
    throw runtime_error(dynslam::utils::Format(
        "Unsupported dataset type: %s", FLAGS_dataset_type.c_str()
    ));
  }

  DynSlam *dyn_slam;
  Input *input;
  BuildDynSlamKittiOdometryGT(dataset_root, &dyn_slam, &input);

  gui::PangolinGui pango_gui(dyn_slam, input);
  pango_gui.Run();

  delete dyn_slam;
  delete input;
}

