
#include <atomic>
#include <chrono>
#include <iomanip>
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
#include "PrecomputedDepthProvider.h"
#include "InstRecLib/VisoSparseSFProvider.h"
#include "DSHandler3D.h"
#include "Evaluation/Evaluation.h"
#include "Evaluation/ErrorVisualizationCallback.h"

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
DEFINE_int32(frame_offset, 0, "The frame index from which to start reading the dataset sequence.");

DEFINE_bool(voxel_decay, true, "Whether to enable map regularization via voxel decay (a.k.a. voxel "
                               "garbage collection).");
DEFINE_int32(min_decay_age, 50, "The minimum voxel *block* age for voxels within it to be eligible "
                                " for deletion (garbage collection).");
DEFINE_int32(max_decay_weight, 2, "The maximum voxle weight for decay. Voxels which have "
                                  "accumulated more than this many measurements will not be "
                                  "removed.");

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
using namespace dynslam::eval;
using namespace dynslam::utils;

static const int kUiWidth = 300;

/// \brief What reconstruction error to visualize.
enum VisualizeError {
  kNone = 0,
  kInputVsLidar,
  kFusionVsLidar,
  kInputVsFusion,
  kEnd
};

/// \brief The main GUI and entry point for DynSLAM.
class PangolinGui {
public:
  PangolinGui(DynSlam *dyn_slam, Input *input)
      : dyn_slam_(dyn_slam),
        dyn_slam_input_(input),
        lidar_vis_colors_(new unsigned char[2500000]),
        lidar_vis_vertices_(new float[2500000])
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
    delete pane_texture_mono_uchar_;

    delete lidar_vis_colors_;
    delete lidar_vis_vertices_;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);
      pangolin::GlFont &font = pangolin::GlFont::I();

      if (autoplay_->Get()) {
        ProcessFrame();
      }

      main_view_->Activate();
      glColor3f(1.0f, 1.0f, 1.0f);

      // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance,
      // so you get dark, uncolored images.

      // Some experimental code for getting the camera to move on its own.
      if (wiggle_mode_->Get()) {
        timeval tp;
        gettimeofday(&tp, nullptr);
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

      if (dyn_slam_->GetCurrentFrameNo() > 1) {
        auto velodyne = dyn_slam_->GetEvaluation()->GetVelodyne();
//        auto lidar_pointcloud = velodyne->ReadFrame(dyn_slam_input_->GetCurrentFrame() - 1);
        auto lidar_pointcloud = velodyne->GetLatestFrame();
        float min_depth_meters = dyn_slam_input_->GetDepthProvider()->GetMinDepthMeters();
        float max_depth_meters = dyn_slam_input_->GetDepthProvider()->GetMaxDepthMeters();

        // TODO(andrei): Pose history in dynslam; will be necessary in delayed evaluation, as well
        // as if you wanna preview cute little frustums!
        Eigen::Matrix4f epose = dyn_slam_->GetPose().inverse();
        auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

        const uchar *synthesized_depthmap = dyn_slam_->GetStaticMapRaycastPreview(
            pango_pose,
//            pane_cam_->GetModelViewMatrix(),
            PreviewType::kDepth
        );
        const cv::Mat1s *input_depthmap = dyn_slam_->GetDepthPreview();
//        cv::Mat1b input_depthmap_uc(height_, width_);

        // TODO(andrei): Don't waste memory...
        uchar input_depthmap_uc[width_ * height_ * 4];

        for(int row = 0; row < height_; ++row) {
          for(int col = 0; col < width_; ++col) {
            // The (signed) short-valued depth map encodes the depth expressed in millimeters.
            short in_depth = input_depthmap->at<short>(row, col);
            // TODO(andrei): Use the affine depth calib params here if needed...
            uchar byte_depth;
            if (in_depth == std::numeric_limits<short>::max()) {
              byte_depth = 0;
            }
            else {
              // TODO(andrei): Eval the impact of rounding, maybe..
              byte_depth = static_cast<uchar>(round(
                  in_depth / 1000.0 / max_depth_meters * 255));
            }

//            input_depthmap_uc.at<uchar>(row, col) = byte_depth;

            int idx = (row * width_ + col) * 4;
            input_depthmap_uc[idx] =     byte_depth;
            input_depthmap_uc[idx + 1] = byte_depth;
            input_depthmap_uc[idx + 2] = byte_depth;
            input_depthmap_uc[idx + 3] = byte_depth;
          }
        }

        uchar diff_buffer[width_ * height_ * 4];
        memset(diff_buffer, '\0', sizeof(uchar) * width_ * height_ * 4);

        const uchar * compare_lidar_vs = nullptr;
        const unsigned char *preview = nullptr;
        int delta_max_visualization = 1;
        string message;
        switch(current_lidar_vis_) {
          case kNone:
            message = "Free cam preview";
            // Render the normal preview with no lidar overlay
            preview = dyn_slam_->GetStaticMapRaycastPreview(
                pane_cam_->GetModelViewMatrix(),
                static_cast<PreviewType>(current_preview_type_));
            pane_texture_->Upload(preview, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            break;
          case kInputVsLidar:
            message = utils::Format("Input depth vs. LIDAR | delta_max = %d", delta_max_visualization);
            compare_lidar_vs = input_depthmap_uc;
            break;
          case kFusionVsLidar:
            message = utils::Format("Fused map vs. LIDAR | delta_max = %d", delta_max_visualization);
            compare_lidar_vs = synthesized_depthmap;
            break;

          case kInputVsFusion:
            message = "Input depth vs. fusion";
            DiffDepthmaps(input_depthmap_uc, synthesized_depthmap, width_, height_,
                          delta_max_visualization, diff_buffer);
            pane_texture_->Upload(diff_buffer, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            break;

          default:
          case kEnd:
            throw runtime_error("Unexpected 'current_lidar_vis_' error visualization mode.");
            break;
        }

        if (compare_lidar_vs != nullptr) {
          pane_texture_->Upload(compare_lidar_vs, GL_RGBA, GL_UNSIGNED_BYTE);
          pane_texture_->RenderToViewport(true);

          bool visualize_input = (current_lidar_vis_ == kInputVsLidar);

          eval::ErrorVisualizationCallback vis_callback(
              delta_max_visualization, visualize_input, Eigen::Vector2f(
                  main_view_->GetBounds().w, main_view_->GetBounds().h), lidar_vis_colors_, lidar_vis_vertices_);

          DepthEvaluation result = dyn_slam_->GetEvaluation()->EvaluateDepth(
              lidar_pointcloud,
              synthesized_depthmap,
              input_depthmap_uc,
              velodyne->velodyne_to_rgb,
              dyn_slam_->GetLeftRgbProjectionMatrix().cast<double>(),
              dyn_slam_->GetRightRgbProjectionMatrix().cast<double>(),
              width_,
              height_,
              min_depth_meters,
              max_depth_meters,
              delta_max_visualization,
              4,
              4,
              &vis_callback
          );
          DepthResult depth_result = current_lidar_vis_ == kFusionVsLidar ? result.fused_result
                                                                          : result.input_result;
          message += utils::Format(" | Acc (with missing): %.3lf | Acc (ignore missing): %.3lf",
                                   depth_result.GetCorrectPixelRatio(true),
                                   depth_result.GetCorrectPixelRatio(false));

          vis_callback.Render();
        }

        font.Text(message).Draw(-0.90f, 0.80f);
      }

      font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);

      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      if(dyn_slam_->GetCurrentFrameNo() >= 1) {
        if (display_raw_previews_->Get()) {
          UploadCvTexture(*(dyn_slam_->GetRgbPreview()), *pane_texture_);
        } else {
          UploadCvTexture(*(dyn_slam_->GetStaticRgbPreview()), *pane_texture_);
        }
        pane_texture_->RenderToViewport(true);

        Tic("LIDAR render");
        auto velodyne = dyn_slam_->GetEvaluation()->GetVelodyne();
        if (velodyne->HasLatestFrame()) {
          PreviewLidar(velodyne->GetLatestFrame(),
                       dyn_slam_->GetLeftRgbProjectionMatrix(),
                       velodyne->velodyne_to_rgb.cast<float>(),
                       rgb_view_);
        }
        else {
          PreviewLidar(velodyne->ReadFrame(dyn_slam_input_->GetCurrentFrame() - 1),
                       dyn_slam_->GetRightRgbProjectionMatrix(),
                       velodyne->velodyne_to_rgb.cast<float>(),
                       rgb_view_);
        }

        Toc(true);
      }

      if (dyn_slam_->GetCurrentFrameNo() > 1 && preview_sf_->Get()) {
        PreviewSparseSF(dyn_slam_->GetLatestFlow().matches, rgb_view_);
      }

      depth_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      // TODO(andrei): Make these rendered buffers use the same color map (currently differs and
      // looks bad when the staticdepthpreview is used).
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
      // TODO(andrei): Make this gradual; currently it's shown as a single-colored blob
//      pane_texture_->Upload(dyn_slam_->GetObjectDepthPreview(visualized_object_idx_),
//                            GL_RED, GL_FLOAT);
      pane_texture_->RenderToViewport(true);

      auto &tracker = dyn_slam_->GetInstanceReconstructor()->GetInstanceTracker();
      if (dyn_slam_->GetCurrentFrameNo() > 0 && preview_sf_->Get()) {
        // TODO-LOW(andrei): This is bonkers. Add some helper methods!
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
      const auto &bbox = latest_detection.copy_mask->GetBoundingBox();
      auto gl_pos = utils::PixelsToGl(Eigen::Vector2f(bbox.r.x0, bbox.r.y0 - font.Height()),
                                      Eigen::Vector2f(width_, height_),
                                      Eigen::Vector2f(segment_view_.GetBounds().w,
                                                      segment_view_.GetBounds().h));

      stringstream info_label;
      info_label << latest_detection.GetClassName() << "#" << track.GetId()
//                 << "@" << setprecision(2) << latest_detection.class_probability
                 << " [" << track.GetStateLabel().substr(0, 1) << "].";
      glColor3f(1.0f, 0.0f, 0.0f);
      font.Text(info_label.str()).Draw(gl_pos[0], gl_pos[1], 0);
    }
  }

  /// \brief Renders a simple preview of the scene flow information onto the currently active pane.
  void PreviewSparseSF(const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow, const pangolin::View &view) {
    pangolin::GlFont &font = pangolin::GlFont::I();
    Eigen::Vector2f frame_size(width_, height_);
    font.Text("libviso2 scene flow preview").Draw(-0.98f, 0.89f);

    // We don't need z-checks since we're rendering UI stuff.
    glDisable(GL_DEPTH_TEST);
    for(const RawFlow &match : flow) {
      Eigen::Vector2f bounds(segment_view_.GetBounds().w, segment_view_.GetBounds().h);
      Eigen::Vector2f gl_pos = PixelsToGl(match.curr_left, frame_size, bounds);
      Eigen::Vector2f gl_pos_old = PixelsToGl(match.prev_left, frame_size, bounds);

      Eigen::Vector2f delta = gl_pos - gl_pos_old;
      float magnitude = 15.0f * static_cast<float>(delta.norm());

      glColor4f(0.3f, 0.3f, 0.9f, 1.0f);
      pangolin::glDrawCross(gl_pos[0], gl_pos[1], 0.025f);
      glColor4f(max(0.2f, min(1.0f, magnitude)), 0.4f, 0.4f, 1.0f);
      pangolin::glDrawLine(gl_pos_old[0], gl_pos_old[1], gl_pos[0], gl_pos[1]);
    }

    glEnable(GL_DEPTH_TEST);
  }

  void DiffDepthmaps(
      const uchar * input_depthmap,
      const uchar * synthesized_depthmap,
      int width,
      int height,
      int delta_max,
      uchar * out_image,
      int stride = 4
  ) {

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int idx = (i * width + j) * stride;
        int input = input_depthmap[idx];
        int synth = synthesized_depthmap[idx];

        if (input == 0 || synth == 0) {
          continue;
        }

        int delta = input - synth;
        int abs_delta = abs(delta);
        if (abs_delta > delta_max) {
          out_image[idx + 0] = min(255, 100 + (abs_delta - delta) * 10);
          out_image[idx + 1] = 0;
          out_image[idx + 2] = 0;
          out_image[idx + 3] = 255;
        }
        else {
          out_image[idx + 0] = 20;
          out_image[idx + 1] = 180;
          out_image[idx + 2] = 20;
          out_image[idx + 3] = 255;
        }
      }
    }

  }

  /*
  DepthResult PreviewError(
    const Eigen::MatrixX4f &lidar_points,
    const Eigen::Matrix<double, 3, 4> &P,
    const Eigen::Matrix4d &Tr,
    const pangolin::View &view,
    const uchar * const computed_depth,
    int width,
    int height,
    float max_depth_meters,
    const int delta_max
  ) {
    // TODO(andrei) XXX reduce code duplication between this and the main eval module
    static GLfloat verts[2000000];
    static GLubyte colors[2000000];

    size_t idx_v = 0;
    size_t idx_c = 0;

    int errors = 0;
    int missing = 0;
    int measurements = 0;
    int correct = 0;

//    long delta_sum = 0;

    glDisable(GL_DEPTH_TEST);
    // Doing this via a loop is much slower, but clearer and less likely to lead to evaluation
    // mistakes than simply rendering the lidar in 2D and comparing it with the map.
    for (int i = 0; i < lidar_points.rows(); ++i) {
      Eigen::Vector4d point = lidar_points.row(i);
      float reflectance = lidar_points(i, 3);
      point(3) = 1.0f;                // Replace reflectance with the homogeneous 1.

      Eigen::Vector4f p3d = Tr * point;
      p3d /= p3d(3);
      float Z = p3d(2);

      // TODO(andrei): Document these limits and explain them in the thesis as well.
      if (Z < 0.5 || Z >= max_depth_meters) {
        continue;
      }

      float Z_scaled = (Z / max_depth_meters) * 255;
      if (Z_scaled <= 0 || Z_scaled > 255) {
        throw runtime_error("Unexpected Z value");
      }

      uchar depth_lidar_uc = static_cast<uchar>(Z_scaled);

      // Note that Sengupta et al. operate on pixels, not individually projected LIDAR points, so it
      // should be OK if we do a 2d rendering of the lidar and compare all nonblack pixels.

      Eigen::VectorXf p2d = P * p3d;
      p2d /= p2d(2);

      if (p2d(0) < 0 || p2d(0) >= width_ || p2d(1) < 0 || p2d(1) >= height_) {
        continue;
      }

      Eigen::Matrix<uchar, 3, 1> color;

      // We should probably do bilinear interpolation here
      int row = static_cast<int>(round(p2d(1)));
      int col = static_cast<int>(round(p2d(0)));
      uchar depth_computed_uc = computed_depth[(row * width_ + col) * 4];

      int delta_signed = static_cast<int>(depth_computed_uc) - static_cast<int>(depth_lidar_uc);

        int delta = abs(delta_signed);
        measurements++;
//        delta_sum += delta_signed;

        // If the delta is larger than the max value OR if we don't have a measurement for the depth
      if(depth_computed_uc == 0) {
        missing++;
      }
      else if (delta > delta_max) {
        errors++;
      }
      else {
        correct++;
      }
        if (delta > delta_max || depth_computed_uc == 0) {

          color(0) = min(255, delta * 10);
          color(1) = 160;
          color(2) = 160;
        } else {
          color(0) = 10;
          color(1) = 255 - delta * 10;
          color(2) = 255 - delta * 10;
//          float intensity = min(8.0f / Z, 1.0f);
//          color(0) = static_cast<uchar>(intensity * 255);
//          color(1) = static_cast<uchar>(intensity * 255);
//          color(2) = static_cast<uchar>(reflectance * 255);
        }
//      }
//      else {
//        continue;
//      }

      Eigen::Vector2f frame_size(width_, height_);
      Eigen::Vector2f gl_pos = PixelsToGl(Eigen::Vector2f(p2d(0), p2d(1)), frame_size, view);

      GLfloat x = gl_pos(0);
      GLfloat y = gl_pos(1);

      verts[idx_v++] = x;
      verts[idx_v++] = y;

      colors[idx_c++] = color(0);
      colors[idx_c++] = color(1);
      colors[idx_c++] = color(2);
    }

    pangolin::glDrawColoredVertices<float>(idx_v / 2, verts, colors, GL_POINTS, 2, 3);
    glEnable(GL_DEPTH_TEST);

    return DepthResult(measurements, errors, missing, correct);
  }
   */

  /// \brief Renders the velodyne points for visual inspection.
  /// \param lidar_points
  /// \param P Left camera matrix.
  /// \param Tr Transforms velodyne points into the left camera's frame.
  /// \note For fused visualization we need to use the depth render as a zbuffer when rendering
  /// LIDAR points, either in OpenGL, or manually by projecting LIDAR points and manually checking
  /// their resulting depth. But we don't need this visualization yet; So far, it's enough to render
  /// the LIDAR results for sanity, and then for every point in the cam frame look up the model
  /// depth and compare the two.
  void PreviewLidar(
      const Eigen::MatrixX4f &lidar_points,
      const Eigen::MatrixXf &P,
      const Eigen::Matrix4f &Tr,
      const pangolin::View &view
  ) {
    // convert every velo point into 2D as: x_i = P * Tr * X_i
    if (lidar_points.rows() == 0) {
      return;
    }
//    static GLfloat verts[2000000];
//    static GLubyte colors[2000000];

    size_t idx_v = 0;
    size_t idx_c = 0;

    glDisable(GL_DEPTH_TEST);
    for (int i = 0; i < lidar_points.rows(); ++i) {
      Eigen::Vector4f point = lidar_points.row(i);
      float reflectance = lidar_points(i, 3);
      point(3) = 1.0f;                // Replace reflectance with the homogeneous 1.

      Eigen::Vector4f p3d = Tr * point;
      p3d /= p3d(3);
      float Z = p3d(2);

      // This part is VERY slow and should be performed in hardware...
//      Eigen::VectorXf p2d = P * p3d;
//      p2d /= p2d(2);

//      if (p2d(0) < 0 || p2d(0) >= width_ || p2d(1) < 0 || p2d(1) >= height_) {
//        continue;
//      }

//      Eigen::Vector2f frame_size(width_, height_);
//      cv::Vec2f gl_pos = PixelsToGl(Eigen::Vector2f(p2d(0), p2d(1)), frame_size, view);
//
//      GLfloat x = gl_pos(0);
//      GLfloat y = gl_pos(1);
//      GLfloat x = p2d(0);
//      GLfloat y = p2d(1);

//      verts[idx_v++] = x;
//      verts[idx_v++] = y;
      lidar_vis_vertices_[idx_v++] = p3d(0);
      lidar_vis_vertices_[idx_v++] = p3d(1);
      lidar_vis_vertices_[idx_v++] = p3d(2);

      float intensity = min(8.0f / Z, 1.0f);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(intensity * 255);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(intensity * 255);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(reflectance * 255);
    }

    float fx = P(0, 0);
    float fy = P(1, 1);
    float cx = P(0, 2);
    float cy = P(1, 2);
    auto proj = pangolin::ProjectionMatrix(width_, height_, fx, fy, cx, cy, 0.01, 1000);

//    auto milliseconds_since_epoch = (
//        std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1) - 1501147873988);
    pangolin::OpenGlRenderState state(
        proj, pangolin::IdentityMatrix().RotateX(M_PI)
    );
    state.Apply();

    // TODO-LOW(andrei): For increased performance (unnecessary), consider just passing ptr to
    // internal eigen data. Make sure its row-major though, and the modelview matrix is set properly
    // based on the velo-to-camera matrix.
    pangolin::glDrawColoredVertices<float>(idx_v / 3, lidar_vis_vertices_, lidar_vis_colors_, GL_POINTS, 3, 3);
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

    /***************************************************************************
     * GUI Buttons
     **************************************************************************/
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
      if (dyn_slam_->GetCurrentFrameNo() < 2) {
        cerr << "Warning: no map to save!" << endl;
      }
      else {
        cout << "Saving static map..." << endl;
        dyn_slam_->SaveStaticMap(dyn_slam_input_->GetSequenceName(),
                                 dyn_slam_input_->GetDepthProvider()->GetName());
        cout << "Mesh generated OK. Writing asynchronously to the disk..." << endl;
        Toc();
      }
    };

    pangolin::Var<function<void(void)>> save_map_button("ui.[S]ave Static Map", save_map);
    pangolin::RegisterKeyPressCallback('s', save_map);
    pangolin::Var<function<void(void)>> reap_button(
        "ui.Force cleanup of current instance",
        [this]() { dyn_slam_->ForceDynamicObjectCleanup(visualized_object_idx_); }
    );

    reconstructions = new pangolin::Var<string>("ui.Rec", "");

    pangolin::Var<function<void(void)>> previous_object("ui.Previous Object [z]", [this]() {
      SelectPreviousVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('z', [this]() { SelectPreviousVisualizedObject(); });
    pangolin::Var<function<void(void)>> next_object("ui.Ne[x]t Object", [this]() {
      SelectNextVisualizedObject();
    });
    pangolin::RegisterKeyPressCallback('x', [this]() { SelectNextVisualizedObject(); });
    auto save_object = [this]() {
      dyn_slam_->SaveDynamicObject(dyn_slam_input_->GetSequenceName(),
                                     dyn_slam_input_->GetDepthProvider()->GetName(),
                                     visualized_object_idx_);
    };
    pangolin::Var<function<void(void)>> save_active_object("ui.Save Active [O]bject", save_object);
    pangolin::RegisterKeyPressCallback('o', save_object);

    auto quit = [this]() {
      dyn_slam_->WaitForJobs();
      pangolin::QuitAll();
    };
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

    pangolin::RegisterKeyPressCallback('0', [&]() {
      if(++current_lidar_vis_ >= VisualizeError::kEnd) {
        current_lidar_vis_ = 0;
      }
    });
    pangolin::RegisterKeyPressCallback('9', [&]() {
      if(--current_lidar_vis_ < 0) {
        current_lidar_vis_ = (VisualizeError::kEnd - 1);
      }
    });

    /***************************************************************************
     * GUI Checkboxes
     **************************************************************************/
    wiggle_mode_ = new pangolin::Var<bool>("ui.Wiggle mode", false, true);
    autoplay_ = new pangolin::Var<bool>("ui.[A]utoplay", false, true);
    pangolin::RegisterKeyPressCallback('a', [this]() {
      *(this->autoplay_) = ! *(this->autoplay_);
    });
    display_raw_previews_ = new pangolin::Var<bool>("ui.Raw Previews", true, true);
    preview_sf_ = new pangolin::Var<bool>("ui.Show Scene Flow", false, true);


    // This is used for the free view camera. The focal lengths are not used in rendering, BUT they
    // impact the sensitivity of the free view camera. The smaller they are, the faster the camera
    // responds to input (ideally, you should use the translation and zoom scales to control this,
    // though).
    float cam_focal_length = 250.0f;
    proj_ = pangolin::ProjectionMatrix(width_, height_,
                                       cam_focal_length, cam_focal_length,
                                       width_ / 2, height_ / 2,
                                       0.1, 1000);

    pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(0, 0, 0,
                                  0, 0, -1,
                                  pangolin::AxisY));
    instance_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAt(
          // -y is up
          0.0, 0.50,  6.75,
          0.0, 0.50,  4.0,
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
    this->pane_texture_mono_uchar_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0,
                                                             GL_RED, GL_UNSIGNED_BYTE);
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
  pangolin::GlTexture *pane_texture_mono_uchar_;

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

  int current_lidar_vis_ = VisualizeError::kFusionVsLidar;

  unsigned char *lidar_vis_colors_;
  float *lidar_vis_vertices_;

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
    glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img_.step / dummy_img_.elemSize());
    dummy_image_texture_->Upload(dummy_img_.data, GL_BGR, GL_UNSIGNED_BYTE);
  }
};

} // namespace gui

Eigen::Matrix<double, 3, 4> ReadProjection(const string &expected_label, istream &in) {
  Eigen::Matrix<double, 3, 4> matrix;
  string label;
  in >> label;
  assert(expected_label == label && "Unexpected token in calibration file.");

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> matrix(row, col);
    }
  }

  return matrix;
};

/// \brief Reads the projection and transformation matrices for a KITTI-odometry sequence.
/// \note P0 = left-gray, P1 = right-gray, P2 = left-color, P3 = right-color
void ReadKittiOdometryCalibration(const string &fpath,
                                  Eigen::Matrix<double, 3, 4> &left_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &right_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &left_color_proj,
                                  Eigen::Matrix<double, 3, 4> &right_color_proj,
                                  Eigen::Matrix4d &velo_to_left_cam) {
  static const string kLeftGray = "P0:";
  static const string kRightGray = "P1:";
  static const string kLeftColor = "P2:";
  static const string kRightColor = "P3:";
  ifstream in(fpath);

  left_gray_proj = ReadProjection(kLeftGray, in);
  right_gray_proj = ReadProjection(kRightGray, in);
  left_color_proj = ReadProjection(kLeftColor, in);
  right_color_proj = ReadProjection(kRightColor, in);

  string dummy;
  in >> dummy;
  assert(dummy == "Tr:");
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> velo_to_left_cam(row, col);
    }
  }
  velo_to_left_cam(3, 0) = 0.0;
  velo_to_left_cam(3, 1) = 0.0;
  velo_to_left_cam(3, 2) = 0.0;
  velo_to_left_cam(3, 3) = 1.0;
}

/// \brief Probes a dataset folder to find the frame dimentsions.
/// \note This is useful for pre-allocating buffers in the rest of the pipeline.
/// \returns A (width, height), i.e., (cols, rows)-style dimension.
Eigen::Vector2i GetFrameSize(const string &dataset_root, const Input::Config &config) {
  string lc_folder = dataset_root + "/" + config.left_color_folder;
  stringstream lc_fpath_ss;
  lc_fpath_ss << lc_folder << "/" << utils::Format(config.fname_format, 1);

  cv::Mat frame = imread(lc_fpath_ss.str());

  return Eigen::Vector2i(
      frame.cols,
      frame.rows
  );
}

// TODO(andrei): Move this to the ITM driver once it's done.
ITMLib::Objects::ITMRGBDCalib* CreateItmCalib(
    const Eigen::Matrix<double, 3, 4> &left_cam_proj,
    const Eigen::Vector2i &frame_size
) {
  ITMRGBDCalib *calib = new ITMRGBDCalib;
  float kMetersToMillimeters = 1.0f / 1000.0f;

  ITMIntrinsics intrinsics;
  float fx = static_cast<float>(left_cam_proj(0, 0));
  float fy = static_cast<float>(left_cam_proj(1, 1));
  float cx = static_cast<float>(left_cam_proj(0, 2));
  float cy = static_cast<float>(left_cam_proj(1, 2));
  float sizeX = frame_size(0);
  float sizeY = frame_size(1);
  intrinsics.SetFrom(fx, fy, cx, cy, sizeX, sizeY);

  // Our intrinsics are always the same for RGB and depth since we compute depth from stereo.
  calib->intrinsics_rgb = intrinsics;
  calib->intrinsics_d = intrinsics;

  // RGB and depth "sensors" are one and the same, so the relative pose is the identity matrix.
  Matrix4f identity; identity.setIdentity();
  calib->trafo_rgb_to_depth.SetFrom(identity);

  // These parameters are used by ITM to convert from the input depth, expressed in millimeters, to
  // the internal depth, which is expressed in meters.
  calib->disparityCalib.SetFrom(kMetersToMillimeters, 0.0f, ITMDisparityCalib::TRAFO_AFFINE);
  return calib;
}

/// \brief Constructs a DynSLAM instance to run on a KITTI Odometry dataset sequence, using the
///        ground truth pose information from the Inertial Navigation System (INS) instead of any
///        visual odometry.
/// XXX: Update docs once you add proper support for selecting GT or libviso2 odometry.
/// This is useful when you want to focus on the quality of the reconstruction, instead of that of
/// the odometry.
void BuildDynSlamKittiOdometryGT(const string &dataset_root, DynSlam **dyn_slam_out, Input **input_out) {

//  Input::Config input_config = Input::KittiOdometryConfig();
  Input::Config input_config = Input::KittiOdometryDispnetConfig();

  Eigen::Matrix34d left_gray_proj;
  Eigen::Matrix34d right_gray_proj;
  Eigen::Matrix34d left_color_proj;
  Eigen::Matrix34d right_color_proj;
  Eigen::Matrix4d velo_to_left_gray_cam;

  // Read all the calibration info we need.
  // HERE BE DRAGONS (for noobs like myself): Make sure you're using the correct matrix for the
  // grayscale and/or color cameras!
  ReadKittiOdometryCalibration(dataset_root + "/" + input_config.calibration_fname,
                               left_gray_proj, right_gray_proj, left_color_proj, right_color_proj,
                               velo_to_left_gray_cam);

  Eigen::Vector2i frame_size = GetFrameSize(dataset_root, input_config);

  cout << "Read calibration from KITTI-style data..." << endl
       << "Proj (left, gray): " << endl << left_gray_proj << endl
       << "Proj (right, gray): " << endl << right_gray_proj << endl
       << "Proj (left, color): " << endl << left_color_proj << endl
       << "Proj (right, color): " << endl << right_color_proj << endl
       << "Velo: " << endl << velo_to_left_gray_cam << endl;

  VoxelDecayParams voxel_decay_params(
      FLAGS_voxel_decay,
      FLAGS_min_decay_age,
      FLAGS_max_decay_weight
  );

  int frame_offset = FLAGS_frame_offset;

  // TODO-LOW(andrei): Compute the baseline from the projection matrices.
  float baseline_m = 0.537150654273f;
  // TODO(andrei): Be aware of which camera you're using for the depth estimation. (for pure focal
  // length it doesn't matter, though, since it's the same)
  float focal_length_px = left_gray_proj(0, 0);
  StereoCalibration stereo_calibration(baseline_m, focal_length_px);

  *input_out = new Input(
      dataset_root,
      input_config,
      nullptr,          // set the depth later
      frame_size,
      stereo_calibration,
      frame_offset);
  DepthProvider *depth = new PrecomputedDepthProvider(
      *input_out,
      dataset_root + "/" + input_config.depth_folder,
      input_config.depth_fname_format,
      input_config.read_depth,
      frame_offset,
      input_config.min_depth_m,
      input_config.max_depth_m
  );
  (*input_out)->SetDepthProvider(depth);

  // [RIP] I lost a couple of hours debugging a bug caused by the fact that InfiniTAM still works
  // even when there is a discrepancy between the size of the depth/rgb inputs, as specified in the
  // calibration file, and the actual size of the input images (but it screws up the previews).

  ITMLibSettings *driver_settings = new ITMLibSettings();
  driver_settings->groundTruthPoseFpath = dataset_root + "/" + input_config.odometry_fname;
  driver_settings->groundTruthPoseOffset = frame_offset;

  drivers::InfiniTamDriver *driver = new InfiniTamDriver(
      driver_settings,
      CreateItmCalib(left_color_proj, frame_size),
      ToItmVec((*input_out)->GetRgbSize()),
      ToItmVec((*input_out)->GetDepthSize()),
      voxel_decay_params);

  const string seg_folder = dataset_root + "/" + input_config.segmentation_folder;
  auto segmentation_provider =
      new instreclib::segmentation::PrecomputedSegmentationProvider(seg_folder, frame_offset);

  VisualOdometryStereo::parameters sf_params;
  // TODO(andrei): The main VO (which we're not using viso2 for, at the moment (June '17) and the
  // "VO" we use to align object instance frames have VASTLY different requirements, so we should
  // use separate parameter sets for them.
  sf_params.base = baseline_m;
  sf_params.match.nms_n = 3;          // Optimal from KITTI leaderboard: 3 (also the default)
  sf_params.match.half_resolution = 0;
  sf_params.match.multi_stage = 1;    // Default = 1 (= 0 => much slower)
  sf_params.match.refinement = 1;     // Default = 1 (per-pixel); 2 = sub-pixel, slower
  sf_params.ransac_iters = 500;       // Default = 200; added more to see if it helps instance reconstruction
  sf_params.inlier_threshold = 3.0;   // Default = 2.0 => we attempt to be coarser for the sake of reconstructing
                                      // object instances
  sf_params.bucket.max_features = 10;    // Default = 2
  // VO is computed using the grayscale frames.
  sf_params.calib.cu = left_gray_proj(0, 2);
  sf_params.calib.cv = left_gray_proj(1, 2);
  sf_params.calib.f  = left_gray_proj(0, 0);

  auto sparse_sf_provider = new instreclib::VisoSparseSFProvider(sf_params);

  auto evaluation = new dynslam::eval::Evaluation(dataset_root, *input_out,
                                                  velo_to_left_gray_cam,
                                                  driver_settings->sceneParams.voxelSize);

  Vector2i input_shape((*input_out)->GetRgbSize().width, (*input_out)->GetRgbSize().height);
  *dyn_slam_out = new gui::DynSlam(
      driver,
      segmentation_provider,
      sparse_sf_provider,
      evaluation,
      input_shape,
      input_config.max_depth_m,
      left_color_proj.cast<float>(),
      right_color_proj.cast<float>()
  );
}

} // namespace dynslam

int main(int argc, char **argv) {
  gflags::SetUsageMessage("The GUI for the DynSLAM dense simultaneous localization and mapping "
                          "system for dynamic environments.\nThis project was built as Andrei "
                          "Barsan's MSc Thesis in Computer Science at the Swiss Federal "
                          "Institute of Technology, Zurich (ETHZ) in the Spring/Summer of 2017, "
                          "under the supervision of Liu Peidong and Professor Andreas Geiger.\n\n"
                          "Project webpage with source code and more information: "
                          "https://github.com/AndreiBarsan/DynSLAM\n\n"
                          "Based on the amazing InfiniTAM volumetric fusion framework (https://github.com/victorprad/InfiniTAM).\n"
                          "Please see the README.md file for further information and credits.");
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

  dynslam::DynSlam *dyn_slam;
  dynslam::Input *input;
  BuildDynSlamKittiOdometryGT(dataset_root, &dyn_slam, &input);

  dynslam::gui::PangolinGui pango_gui(dyn_slam, input);
  pango_gui.Run();

  delete dyn_slam;
  delete input;
}

