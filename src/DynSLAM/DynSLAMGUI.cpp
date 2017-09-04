
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <backward.hpp>
#include <gflags/gflags.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <pangolin/pangolin.h>

#include "DynSlam.h"
#include "PrecomputedDepthProvider.h"
#include "InstRecLib/VisoSparseSFProvider.h"
#include "DSHandler3D.h"
#include "Evaluation/Evaluation.h"
#include "Evaluation/ErrorVisualizationCallback.h"
#include "Evaluation/EvaluationCallback.h"
#include "Evaluation/SegmentedVisualizationCallback.h"

const std::string kKittiOdometry = "kitti-odometry";
const std::string kKittiTracking = "kitti-tracking";
const std::string kKitti         = "kitti";

DEFINE_string(dataset_type,
              kKittiOdometry,
              "The type of the input dataset at which 'dataset_root' is pointing. Supported are "
              "'kitti-odometry' and 'kitti-tracking'.");
DEFINE_string(dataset_root, "", "The root folder of the dataset or dataset sequence to use.");
DEFINE_bool(dynamic_mode, true, "Whether DynSLAM should be aware of dynamic objects and attempt to "
                                "reconstruct them. Disabling this makes the system behave like a "
                                "vanilla outdoor InfiniTAM.");
DEFINE_int32(frame_offset, 0, "The frame index from which to start reading the dataset sequence.");
DEFINE_int32(frame_limit, 0, "How many frames to process in auto mode. 0 = no limit.");
DEFINE_bool(voxel_decay, true, "Whether to enable map regularization via voxel decay (a.k.a. voxel "
                               "garbage collection).");
DEFINE_int32(min_decay_age, 200, "The minimum voxel *block* age for voxels within it to be eligible "
                                "for deletion (garbage collection).");
DEFINE_int32(max_decay_weight, 1, "The maximum voxel weight for decay. Voxels which have "
                                  "accumulated more than this many measurements will not be "
                                  "removed.");
DEFINE_int32(kitti_tracking_sequence_id, -1, "Used in conjunction with --dataset_type kitti-tracking.");
DEFINE_bool(direct_refinement, false, "Whether to refine motion estimates for other cars computed "
                                     "sparsely with RANSAC using a semidense direct image "
                                     "alignment method.");
// TODO-LOW(andrei): Automatically adjust the voxel GC params when depth weighting is enabled.
DEFINE_bool(use_depth_weighting, false, "Whether to adaptively set fusion weights as a function of "
                                        "the inverse depth (w \\propto \\frac{1}{Z}). If disabled, "
                                        "all new measurements have a constant weight of 1.");
DEFINE_bool(semantic_evaluation, true, "Whether to separately evaluate the static and dynamic "
                                       "parts of the reconstruction, based on the semantic "
                                       "segmentation of each frame.");
DEFINE_double(scale, 1.0, "Whether to run in reduced-scale mode. Used for experimental purposes. "
                          "Requires the (odometry) sequence to have been preprocessed using the "
                          "'scale_sequence.py' script.");
DEFINE_bool(use_dispnet, false, "Whether to use DispNet depth maps. Otherwise ELAS is used.");
DEFINE_int32(evaluation_delay, 0, "How many frames behind the current one should the evaluation be "
                                  "performed. A value of 0 signifies always computing the "
                                  "evaluation metrics on the most recent frames. Useful for "
                                  "measuring the impact of the regularization, which ``follows'' "
                                  "the camera with a delay of 'min_decay_age'. Warning: does not "
                                  "support dynamic scenes.");
DEFINE_bool(close_on_complete, true, "Whether to shut down automatically once 'frame_limit' is "
                                     "reached.");
DEFINE_bool(record, false, "Whether to record a video of the GUI and save it to disk. Using an "
                           "external program usually leads to better results, though.");
DEFINE_bool(chase_cam, false, "Whether to preview the reconstruction in chase cam mode, following "
                             "the camera from a third person view.");
DEFINE_int32(fusion_every, 1, "Fuse every kth frame into the map. Used for evaluating the system's "
                              "behavior under reduced temporal resolution.");
DEFINE_bool(autoplay, false, "Whether to start with autoplay enabled. Useful for batch experiments.");

// Note: the [RIP] tags signal spots where I wasted more than 30 minutes debugging a small, silly
// issue, which could easily be avoided in the future.

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
backward::SignalHandling sh;

namespace dynslam {
namespace gui {

using namespace std;
using namespace instreclib::reconstruction;
using namespace instreclib::segmentation;

using namespace dynslam;
using namespace dynslam::eval;
using namespace dynslam::utils;

static const int kUiWidth = 300;

/// \brief What reconstruction error to visualize (used for inspecting the evaluation).
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
        width_(dyn_slam->GetInputWidth()),
        height_(dyn_slam->GetInputHeight()),
        depth_preview_buffer_(dyn_slam->GetInputHeight(), dyn_slam->GetInputWidth()),
        lidar_vis_colors_(new unsigned char[2500000]),
        lidar_vis_vertices_(new float[2500000])
  {
    CreatePangolinDisplays();
  }

  PangolinGui(const PangolinGui&) = delete;
  PangolinGui(PangolinGui&&) = delete;
  PangolinGui& operator=(const PangolinGui&) = delete;
  PangolinGui& operator=(PangolinGui&&) = delete;

  virtual ~PangolinGui() {
    // No need to delete any view pointers; Pangolin deletes those itself on shutdown.
    delete pane_texture_;
    delete pane_texture_mono_uchar_;

    delete lidar_vis_colors_;
    delete lidar_vis_vertices_;
  }

  /// \brief Renders the camera poses as frustums over the free camera view raycast produced by
  ///        InfiniTAM.
  /// \param current_time_ms Used for making the latest pose easier to spot by pulsating its color.
  void DrawPoses(long current_time_ms) {
    main_view_->Activate();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glMatrixMode(GL_PROJECTION);
    proj_.Load();

    glMatrixMode(GL_MODELVIEW);
    pane_cam_->GetModelViewMatrix().Load();

    auto phist = dyn_slam_->GetPoseHistory();

    // Make the poses a little bit more visible (set to > 0.0f to enable).
    float frustum_root_cube_scale = 0.00f;

    const float kMaxFrustumScale = 0.66;
    Eigen::Vector3f color_white(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < static_cast<int>(phist.size()) - 1; ++i) {
      float frustum_scale = max(0.15f, kMaxFrustumScale - 0.05f * (phist.size() - 1 - i));
      DrawPoseFrustum(phist[i], color_white, frustum_scale, frustum_root_cube_scale);
    }

    if (! phist.empty()) {
      // Highlight the most recent pose.
      Eigen::Vector3f glowing_green(
          0.5f, 0.5f + static_cast<float>(sin(current_time_ms / 250.0) * 0.5 + 0.5) * 0.5f, 0.5f);
      DrawPoseFrustum(phist[phist.size() - 1], glowing_green, kMaxFrustumScale, frustum_root_cube_scale);
    }
  }

  void DrawPoseFrustum(const Eigen::Matrix4f &pose, const Eigen::Vector3f &color,
                       float frustum_scale, float frustum_root_cube_scale) const {
    glPushMatrix();
    Eigen::Matrix4f inv_pose = pose.inverse();
    glMultMatrixf(inv_pose.data());
    pangolin::glDrawColouredCube(-frustum_root_cube_scale, frustum_root_cube_scale);
    glPopMatrix();

    Eigen::Matrix34f projection = dyn_slam_->GetLeftRgbProjectionMatrix();
    const Eigen::Matrix3f Kinv = projection.block(0, 0, 3, 3).inverse();
    glColor3f(color(0), color(1), color(2));
    pangolin::glDrawFrustrum(Kinv, width_, height_, inv_pose, frustum_scale);
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);
      pangolin::GlFont &font = pangolin::GlFont::I();

      if (autoplay_->Get()) {
        if (FLAGS_frame_limit == 0 || dyn_slam_->GetCurrentFrameNo() < FLAGS_frame_limit) {
          ProcessFrame();
        }
        else {
          cerr << "Warning: reached autoplay limit of [" << FLAGS_frame_limit << "]. Stopped."
               << endl;
          *autoplay_ = false;
          if (FLAGS_close_on_complete) {
            cerr << "Closing as instructed. Bye!" << endl;
            pangolin::QuitAll();
            return;
          }
        }
      }

      long time_ms = utils::GetTimeMs();

      main_view_->Activate(*pane_cam_);
      glEnable(GL_DEPTH_TEST);
      glColor3f(1.0f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(false);

      if (FLAGS_chase_cam) {
        Eigen::Matrix4f cam_mv = dyn_slam_->GetPose().inverse();
        pangolin::OpenGlMatrix pm(cam_mv);
        pm =
            // Good for odo 05
             pangolin::OpenGlMatrix::RotateY(M_PI * 0.5 * 0.05f) *
            // Good for tracking 02
//             pangolin::OpenGlMatrix::RotateY(-M_PI * 0.5 * 0.20f) *

             pangolin::OpenGlMatrix::RotateX(M_PI * 0.5 * 0.03f) *
            // Good for odo 05
//             pangolin::OpenGlMatrix::Translate(-0.5, 1.2, 20.0) *
            // Good for tracking 02
//             pangolin::OpenGlMatrix::Translate(1.5, 1.2, 12.0) *


             pangolin::OpenGlMatrix::Translate(-0.5, 1.0, 15.0) *
             pm;
        pane_cam_->SetModelViewMatrix(pm);
      }

      int evaluated_frame_idx = dyn_slam_->GetCurrentFrameNo() - 1 - FLAGS_evaluation_delay;
      if (evaluated_frame_idx > 0) {
        auto velodyne = dyn_slam_->GetEvaluation()->GetVelodyneIO();
        int input_frame_idx = dyn_slam_input_->GetFrameOffset() + evaluated_frame_idx;

        Eigen::Matrix4f epose = dyn_slam_->GetPoseHistory()[evaluated_frame_idx + 1];
        auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

        bool enable_compositing = (FLAGS_evaluation_delay == 0);
        const float *synthesized_depthmap = dyn_slam_->GetStaticMapRaycastDepthPreview(pango_pose, enable_compositing);
        auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
        auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
        dyn_slam_input_->GetFrameCvImages(input_frame_idx, input_rgb, input_depthmap);

        /// Result of diffing our disparity maps (input and synthesized).
        uchar diff_buffer[width_ * height_ * 4];
        memset(diff_buffer, '\0', sizeof(uchar) * width_ * height_ * 4);

        bool need_lidar = false;
        const unsigned char *preview = nullptr;
        const uint delta_max_visualization = 1;
        string message;
        switch(current_lidar_vis_) {
          case kNone:
            if (FLAGS_chase_cam) {
              message = "Chase cam preview";
            }
            else {
              message = "Free cam preview";
            }
            preview = dyn_slam_->GetStaticMapRaycastPreview(
                pane_cam_->GetModelViewMatrix(),
//                pango_pose,
                static_cast<PreviewType>(current_preview_type_),
                enable_compositing);
            pane_texture_->Upload(preview, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            DrawPoses(time_ms);
            break;

          case kInputVsLidar:
            message = utils::Format("Input depth vs. LIDAR | delta_max = %d", delta_max_visualization);
            need_lidar = true;
            UploadCvTexture(*input_depthmap, *pane_texture_, false, GL_SHORT);
            break;

          case kFusionVsLidar:
            message = utils::Format("Fused map vs. LIDAR | delta_max = %d", delta_max_visualization);
            need_lidar = true;
            FloatDepthmapToShort(synthesized_depthmap, depth_preview_buffer_);
            UploadCvTexture(depth_preview_buffer_, *pane_texture_, false, GL_SHORT);
            break;

          case kInputVsFusion:
            message = "Input depth vs. fusion (green = OK, yellow = input disp > fused, cyan = input disp < fused";
            DiffDepthmaps(*input_depthmap, synthesized_depthmap, width_, height_,
                          delta_max_visualization, diff_buffer, dyn_slam_->GetStereoBaseline(),
                          dyn_slam_->GetLeftRgbProjectionMatrix()(0, 0));
            pane_texture_->Upload(diff_buffer, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            break;

          default:
          case kEnd:
            throw runtime_error("Unexpected 'current_lidar_vis_' error visualization mode.");
            break;
        }

        // Ensures we have a blank slate for the pane's overlay text.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1.0f, 1.0f, 1.0f);
        main_view_->Activate();

        if (need_lidar) {
          pane_texture_->RenderToViewport(true);
          bool visualize_input = (current_lidar_vis_ == kInputVsLidar);
//          eval::ErrorVisualizationCallback vis_callback(
//              delta_max_visualization,
//              visualize_input,
//              Eigen::Vector2f(main_view_->GetBounds().w, main_view_->GetBounds().h),
//              lidar_vis_colors_,
//              lidar_vis_vertices_);
          auto vis_mode = eval::SegmentedCallback::LidarAssociation::kStaticMap;
          auto seg = dyn_slam_->GetSpecificSegmentationForEval(input_frame_idx);
          eval::SegmentedVisualizationCallback vis_callback(
              delta_max_visualization,
              visualize_input,
              Eigen::Vector2f(main_view_->GetBounds().w, main_view_->GetBounds().h),
              lidar_vis_colors_,
              lidar_vis_vertices_,
              seg.get(),
//              dyn_slam_->GetInstanceReconstructor(),
              nullptr,
              vis_mode
          );
          if (vis_mode == eval::SegmentedCallback::LidarAssociation::kDynamicReconstructed) {
            message += " | Reconstructed dynamic objects only ";
          }
          else if (vis_mode == eval::SegmentedCallback::LidarAssociation::kStaticMap) {
            message += " | Static map only";
          }

          bool compare_on_intersection = true;
          bool kitti_style = true;
          eval::EvaluationCallback eval_callback(delta_max_visualization,
                                                 compare_on_intersection,
                                                 kitti_style);

          if (velodyne->FrameAvailable(input_frame_idx)) {
            auto visualized_lidar_pointcloud = velodyne->ReadFrame(input_frame_idx);
            dyn_slam_->GetEvaluation()->EvaluateDepth(visualized_lidar_pointcloud,
                                                      synthesized_depthmap,
                                                      *input_depthmap,
                                                      {&vis_callback, &eval_callback});
            auto result = eval_callback.GetEvaluation();
            DepthResult depth_result = current_lidar_vis_ == kFusionVsLidar ? result.fused_result
                                                                            : result.input_result;
            message += utils::Format(" | Acc (with missing): %.3lf | Acc (ignore missing): %.3lf",
                                     depth_result.GetCorrectPixelRatio(true),
                                     depth_result.GetCorrectPixelRatio(false));
            vis_callback.Render();
          }
        }

        font.Text(message).Draw(-0.90f, 0.80f);
      }
      //*/
      font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);

      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      if(dyn_slam_->GetCurrentFrameNo() >= 1) {
        if (display_raw_previews_->Get()) {
          UploadCvTexture(*(dyn_slam_->GetRgbPreview()), *pane_texture_, true, GL_UNSIGNED_BYTE);
        } else {
          UploadCvTexture(*(dyn_slam_->GetStaticRgbPreview()), *pane_texture_, true, GL_UNSIGNED_BYTE);
        }
        pane_texture_->RenderToViewport(true);

        Tic("LIDAR render");
//        auto velodyne = dyn_slam_->GetEvaluation()->GetVelodyneIO();
//        if (velodyne->HasLatestFrame()) {
//          PreviewLidar(velodyne->GetLatestFrame(),
//                       dyn_slam_->GetLeftRgbProjectionMatrix(),
//                       dyn_slam_->GetEvaluation()->velo_to_left_gray_cam_.cast<float>(),
//                       rgb_view_);
//        }
//        else {
//          PreviewLidar(velodyne->ReadFrame(dyn_slam_input_->GetCurrentFrame() - 1),
//                       dyn_slam_->GetLeftRgbProjectionMatrix(),
//                       dyn_slam_->GetEvaluation()->velo_to_left_gray_cam_.cast<float>(),
//                       rgb_view_);
//        }
        Toc(true);
      }

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
        UploadCvTexture(*dyn_slam_->GetSegmentationPreview(), *pane_texture_, true, GL_UNSIGNED_BYTE);
        pane_texture_->RenderToViewport(true);
        DrawInstanceLables();
      }

      object_view_.Activate();
      glColor4f(1.0, 1.0, 1.0, 1.0f);
      // Preview instance frame RGB
      {
      pane_texture_->Upload(dyn_slam_->GetObjectPreview(visualized_object_idx_),
                            GL_RGBA, GL_UNSIGNED_BYTE);
      }
      // Preview instance frame depth
      {
//        FloatDepthmapToShort(
//            dyn_slam_->GetObjectDepthPreview(visualized_object_idx_),
//            this->depth_preview_buffer_);
//        UploadCvTexture(this->depth_preview_buffer_, *pane_texture_, false, GL_SHORT);
      }
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

      // TODO(andrei): Re-enable once you finish with the automated experiments.
      // Disable autoplay once we reach the end of a sequence.
//      if (! this->dyn_slam_input_->HasMoreImages()) {
//        (*this->autoplay_) = false;
//      }

      // Swap frames and Process Events
      pangolin::FinishFrame();

      if (FLAGS_record) {
        const string kRecordingRoot = "../recordings/";
        if (! utils::FileExists(kRecordingRoot)) {
          throw std::runtime_error(utils::Format(
              "Recording enabled but the output directory (%s) could not be found!",
              kRecordingRoot.c_str()));
        }
        string frame_fname = utils::Format("recorded-frame-%04d", dyn_slam_->GetCurrentFrameNo());
        pangolin::SaveWindowOnRender(kRecordingRoot + "/" + frame_fname);
      }

    }
  }

  /// \brief Renders informative labels for the currently active track.
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
                 << " [" << track.GetStateLabel().substr(0, 1) << "].";
      glColor3f(1.0f, 0.0f, 0.0f);
      font.Text(info_label.str()).Draw(gl_pos[0], gl_pos[1], 0);
    }
  }

  /// \brief Renders a simple preview of the scene flow information onto the currently active pane.
  void PreviewSparseSF(const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow, const pangolin::View &view) {
    pangolin::GlFont &font = pangolin::GlFont::I();
    Eigen::Vector2f frame_size(width_, height_);
//    font.Text("libviso2 scene flow preview").Draw(-0.90f, 0.89f);

    // We don't need z-checks since we're rendering UI stuff.
    glDisable(GL_DEPTH_TEST);
    for(const RawFlow &match : flow) {
      Eigen::Vector2f bounds(segment_view_.GetBounds().w, segment_view_.GetBounds().h);

      // Very hacky way of making the lines thicker
      for (int xof = -1; xof <= 1; ++xof) {
        for (int yof = -1; yof <= 1; ++yof) {
          Eigen::Vector2f of(xof, yof);
          Eigen::Vector2f gl_pos = PixelsToGl(match.curr_left + of, frame_size, bounds);
          Eigen::Vector2f gl_pos_old = PixelsToGl(match.prev_left + of, frame_size, bounds);

          Eigen::Vector2f delta = gl_pos - gl_pos_old;
          float magnitude = 15.0f * static_cast<float>(delta.norm());

          glColor4f(max(0.2f, min(1.0f, magnitude)), 0.4f, 0.4f, 1.0f);
          pangolin::glDrawCircle(gl_pos.cast<double>(), 0.010f);
          pangolin::glDrawLine(gl_pos_old[0], gl_pos_old[1], gl_pos[0], gl_pos[1]);
        }
      }
    }

    glEnable(GL_DEPTH_TEST);
  }

  /// \brief Produces a visual pixelwise diff image of the supplied depth maps, into out_image.
  void DiffDepthmaps(
      const cv::Mat1s &input_depthmap,
      const float* rendered_depth,
      int width,
      int height,
      int delta_max,
      uchar * out_image,
      float baseline_m,
      float focal_length_px
  ) {

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int in_idx = (i * width + j);
        int out_idx = (i * width + j) * 4;
        float input_depth_m = input_depthmap.at<short>(i, j) / 1000.0f;
        float rendered_depth_m = rendered_depth[in_idx];

        float input_disp = baseline_m * focal_length_px / input_depth_m;
        float rendered_disp = baseline_m * focal_length_px / rendered_depth_m;

        if (input_depth_m == 0 || fabs(rendered_depth_m < 1e-5)) {
          continue;
        }

        float delta = input_disp - rendered_disp;
        float abs_delta = fabs(delta);
        if (abs_delta > delta_max) {
          // Visualize SIGNED delta to highlight areas where a particular method tends to
          // consistently over/underestimate.
          if (delta > 0) {
            out_image[out_idx + 0] = 0;
            out_image[out_idx + 1] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 2] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
          }
          else {
            out_image[out_idx + 0] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 1] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 2] = 0;
          }
        }
        else {
          out_image[out_idx + 0] = 0;
          out_image[out_idx + 1] = 255;
          out_image[out_idx + 2] = 0;
        }
      }
    }

  }

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
      const Eigen::Matrix34f &P,
      const Eigen::Matrix4f &Tr,
      const pangolin::View &view
  ) {
    // convert every velo point into 2D as: x_i = P * Tr * X_i
    if (lidar_points.rows() == 0) {
      return;
    }
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
        dyn_slam_->SaveStaticMap(dyn_slam_input_->GetDatasetIdentifier(),
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
      dyn_slam_->SaveDynamicObject(dyn_slam_input_->GetDatasetIdentifier(),
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

    pangolin::Var<function<void(void)>> collect("ui.Map Voxel [G]C Catchup", [&]() {
      dyn_slam_->StaticMapDecayCatchup();
    });
    pangolin::RegisterKeyPressCallback('g', [&]() { dyn_slam_->StaticMapDecayCatchup(); });

    /***************************************************************************
     * GUI Checkboxes
     **************************************************************************/
    autoplay_ = new pangolin::Var<bool>("ui.[A]utoplay", FLAGS_autoplay, true);
    pangolin::RegisterKeyPressCallback('a', [this]() {
      *(this->autoplay_) = ! *(this->autoplay_);
    });
    display_raw_previews_ = new pangolin::Var<bool>("ui.Raw Previews", false, true);
    preview_sf_ = new pangolin::Var<bool>("ui.Show Scene Flow", false, true);

    pangolin::RegisterKeyPressCallback('r', [&]() {
      *display_raw_previews_ = !display_raw_previews_->Get();
    });

    // This constructs an OpenGL projection matrix from a calibrated camera pinhole projection
    // matrix. They are quite different, and the conversion between them is nontrivial.
    // See https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/ for more info.
    const Eigen::Matrix34f real_cam_proj = dyn_slam_->GetLeftRgbProjectionMatrix();
    float near = 0.01;
    float far = 1000.0f;
    // -y is up
    proj_ = pangolin::ProjectionMatrixRDF_TopLeft(width_, height_,
                                                  real_cam_proj(0, 0), real_cam_proj(1, 1),
                                                  real_cam_proj(0, 2), real_cam_proj(1, 2),
                                                  near, far);

    pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAtRDF(0,  -1.5, 15,
                                     0,  -1.5, 50,
                                     0, 1, 0));
    instance_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAtRDF(
          -0.8, -0.20,  -3,
          -0.8, -0.20,  15,
          0, 1, 0)
    );

    float aspect_ratio = static_cast<float>(width_) / height_;
    rgb_view_ = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view_ = pangolin::Display("depth").SetAspect(aspect_ratio);

    segment_view_ = pangolin::Display("segment").SetAspect(aspect_ratio);
    object_view_ = pangolin::Display("object").SetAspect(aspect_ratio);
    float camera_translation_scale = 1.0f;
    float camera_zoom_scale = 1.0f;

    object_reconstruction_view_ = pangolin::Display("object_3d").SetAspect(aspect_ratio)
        .SetHandler(new DSHandler3D(
            instance_cam_,
            pangolin::AxisY,
            camera_translation_scale,
            camera_zoom_scale
        ));

    // These objects remain under Pangolin's management, so they don't need to be deleted by the
    // current class.
    main_view_ = &(pangolin::Display("main").SetAspect(aspect_ratio));
    main_view_->SetHandler(
        new DSHandler3D(pane_cam_,
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

    if (! dyn_slam_input_->HasMoreImages() && FLAGS_close_on_complete) {
      cerr << "No more images, and I'm instructed to shut down when that happens. Bye!" << endl;
      pangolin::QuitAll();
      return;
    }

    size_t free_gpu_memory_bytes;
    size_t total_gpu_memory_bytes;
    cudaMemGetInfo(&free_gpu_memory_bytes, &total_gpu_memory_bytes);

    const double kBytesToGb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    double free_gpu_gb = static_cast<float>(free_gpu_memory_bytes) * kBytesToGb;
    data_log_.Log(
        active_object_count_,
        static_cast<float>(free_gpu_gb) * 10.0f,   // Mini-hack to make the scales better
        dyn_slam_->GetStaticMapMemoryBytes() * 10.0f * kBytesToGb,
        (dyn_slam_->GetStaticMapMemoryBytes() + dyn_slam_->GetStaticMapSavedDecayMemoryBytes()) * 10.0f * kBytesToGb
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

  static void DrawOutlinedText(cv::Mat &target, const string &text, int x, int y, float scale = 1.5f) {
   int thickness = static_cast<int>(round(1.1 * scale));
   int outline_factor = 3;
   cv::putText(target, text, cv::Point_<int>(x, y),
               cv::FONT_HERSHEY_DUPLEX, scale, cv::Scalar(0, 0, 0), outline_factor * thickness, CV_AA);
   cv::putText(target, text, cv::Point_<int>(x, y),
               cv::FONT_HERSHEY_DUPLEX, scale, cv::Scalar(230, 230, 230), thickness, CV_AA);
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

  pangolin::GlTexture *pane_texture_;
  pangolin::GlTexture *pane_texture_mono_uchar_;

  pangolin::Var<string> *reconstructions;

  // Atomic because it gets set from a UI callback. Technically, Pangolin shouldn't invoke callbacks
  // from a different thread, but using atomics for this is generally a good practice anyway.
  atomic<int> active_object_count_;

  /// \brief When this is on, the input gets processed as fast as possible, without requiring any
  /// user input.
  pangolin::Var<bool> *autoplay_;
  /// \brief Whether to display the RGB and depth previews directly from the input, or from the
  /// static scene, i.e., with the dynamic objects removed.
  pangolin::Var<bool> *display_raw_previews_;
  /// \brief Whether to preview the sparse scene flow on the input and current instance RGP panes.
  pangolin::Var<bool> *preview_sf_;

  // TODO(andrei): Reset button.

  // Indicates which object is currently being visualized in the GUI.
  int visualized_object_idx_ = 0;

  int current_preview_type_ = kColor;

  int current_lidar_vis_ = VisualizeError::kNone;

  cv::Mat1s depth_preview_buffer_;

  unsigned char *lidar_vis_colors_;
  float *lidar_vis_vertices_;

  /// \brief Prepares the contents of an OpenCV Mat object for rendering with Pangolin (OpenGL).
  /// Does not actually render the texture.
  static void UploadCvTexture(
      const cv::Mat &mat,
      pangolin::GlTexture &texture,
      bool color,
      GLenum data_type
  ) {
    int old_alignment, old_row_length;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_alignment);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &old_row_length);

    int new_alignment = (mat.step & 3) ? 1 : 4;
    int new_row_length = static_cast<int>(mat.step / mat.elemSize());
    glPixelStorei(GL_UNPACK_ALIGNMENT, new_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, new_row_length);

    // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance,
    // so you get dark, uncolored images.
    GLenum data_format = (color) ? GL_BGR : GL_LUMINANCE;
    texture.Upload(mat.data, data_format, data_type);

    glPixelStorei(GL_UNPACK_ALIGNMENT, old_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, old_row_length);
  }
};

} // namespace gui

Eigen::Matrix<double, 3, 4> ReadProjection(const string &expected_label, istream &in, double downscale_factor) {
  Eigen::Matrix<double, 3, 4> matrix;
  string label;
  in >> label;
  assert(expected_label == label && "Unexpected token in calibration file.");

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> matrix(row, col);
    }
  }

// The downscale factor is used to adjust the intrinsic matrix for low-res input.
//  cout << "Adjusting projection matrix for scale [" << downscale_factor << "]." << endl;
//  matrix *= downscale_factor;
//  matrix(2, 2) = 1.0;

  return matrix;
};

/// \brief Reads the projection and transformation matrices for a KITTI-odometry sequence.
/// \note P0 = left-gray, P1 = right-gray, P2 = left-color, P3 = right-color
void ReadKittiOdometryCalibration(const string &fpath,
                                  Eigen::Matrix<double, 3, 4> &left_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &right_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &left_color_proj,
                                  Eigen::Matrix<double, 3, 4> &right_color_proj,
                                  Eigen::Matrix4d &velo_to_left_cam,
                                  double downscale_factor) {
  static const string kLeftGray = "P0:";
  static const string kRightGray = "P1:";
  static const string kLeftColor = "P2:";
  static const string kRightColor = "P3:";
  ifstream in(fpath);
  if (! in.is_open()) {
    throw runtime_error(utils::Format("Could not open calibration file: [%s]", fpath.c_str()));
  }

  left_gray_proj = ReadProjection(kLeftGray, in, downscale_factor);
  right_gray_proj = ReadProjection(kRightGray, in, downscale_factor);
  left_color_proj = ReadProjection(kLeftColor, in, downscale_factor);
  right_color_proj = ReadProjection(kRightColor, in, downscale_factor);

  string dummy;
  in >> dummy;
  if (dummy != "Tr:") {
    // Looks like a kitti-tracking sequence
    std::getline(in, dummy); // skip to the end of current line

    in >> dummy;
    assert(dummy == "Tr_velo_cam");
  }

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

  // TODO(andrei): Make this a little nicer if it works.
  cv::Mat frame = cv::imread(lc_fpath_ss.str());
  return Eigen::Vector2i(
      frame.cols * 1.0f / FLAGS_scale,
      frame.rows * 1.0f / FLAGS_scale
  );
}

/// \brief Constructs a DynSLAM instance to run on a KITTI Odometry dataset sequence, using liviso2
///        for visual odometry.
void BuildDynSlamKittiOdometry(const string &dataset_root,
                               DynSlam **dyn_slam_out,
                               Input **input_out) {
  Input::Config input_config;
  double downscale_factor = FLAGS_scale;
  if (downscale_factor > 1.0 || downscale_factor <= 0.0) {
    throw runtime_error("Scaling factor must be > 0 and <= 1.0.");
  }
  float downscale_factor_f = static_cast<float>(downscale_factor);

  if (FLAGS_dataset_type == kKittiOdometry) {
    if (downscale_factor != 1.0) {
      if (FLAGS_use_dispnet) {
        input_config = Input::KittiOdometryDispnetLowresConfig(downscale_factor_f);
      }
      else {
        input_config = Input::KittiOdometryLowresConfig(downscale_factor_f);
      }
    }
    else {
      if (FLAGS_use_dispnet) {
        input_config = Input::KittiOdometryDispnetConfig();
      }
      else {
        input_config = Input::KittiOdometryConfig();
      }
    }
  }
  else if (FLAGS_dataset_type == kKittiTracking){
    int t_seq_id = FLAGS_kitti_tracking_sequence_id;
    if (t_seq_id < 0) {
      throw runtime_error("Please specify a KITTI tracking sequence ID.");
    }

    if (FLAGS_use_dispnet) {
      input_config = Input::KittiTrackingDispnetConfig(t_seq_id);
    }
    else {
      input_config = Input::KittiTrackingConfig(t_seq_id);
    }
  }
  else {
    throw runtime_error(utils::Format("Unknown dataset type: [%s]", FLAGS_dataset_type.c_str()));
  }

  Eigen::Matrix34d left_gray_proj;
  Eigen::Matrix34d right_gray_proj;
  Eigen::Matrix34d left_color_proj;
  Eigen::Matrix34d right_color_proj;
  Eigen::Matrix4d velo_to_left_gray_cam;

  // Read all the calibration info we need.
  // HERE BE DRAGONS Make sure you're using the correct matrix for the grayscale and/or color cameras!
  ReadKittiOdometryCalibration(dataset_root + "/" + input_config.calibration_fname,
                               left_gray_proj, right_gray_proj, left_color_proj, right_color_proj,
                               velo_to_left_gray_cam, downscale_factor);

  Eigen::Vector2i frame_size = GetFrameSize(dataset_root, input_config);

  cout << "Read calibration from KITTI-style data..." << endl
       << "Frame size: " << frame_size << endl
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
      nullptr,          // set the depth provider later
      frame_size,
      stereo_calibration,
      frame_offset,
      downscale_factor);
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
  if (FLAGS_dynamic_weights) {
    driver_settings->sceneParams.maxW = driver_settings->maxWDynamic;
  }

  drivers::InfiniTamDriver *driver = new InfiniTamDriver(
      driver_settings,
      CreateItmCalib(left_color_proj, frame_size),
      ToItmVec((*input_out)->GetRgbSize()),
      ToItmVec((*input_out)->GetDepthSize()),
      voxel_decay_params,
      FLAGS_use_depth_weighting);

  const string seg_folder = dataset_root + "/" + input_config.segmentation_folder;
  auto segmentation_provider =
      new instreclib::segmentation::PrecomputedSegmentationProvider(
          seg_folder, frame_offset, static_cast<float>(downscale_factor));

  VisualOdometryStereo::parameters sf_params;
  // TODO(andrei): The main VO (which we're not using viso2 for, at the moment (June '17) and the
  // "VO" we use to align object instance frames have VASTLY different requirements, so we should
  // use separate parameter sets for them.
  sf_params.base = baseline_m;
  sf_params.match.nms_n = 3;          // Optimal from KITTI leaderboard: 3 (also the default)
  sf_params.match.half_resolution = 0;
  sf_params.match.multi_stage = 1;    // Default = 1 (= 0 => much slower)
  sf_params.match.refinement = 1;     // Default = 1 (per-pixel); 2 = sub-pixel, slower
  sf_params.ransac_iters = 500;       // Default = 200
  sf_params.inlier_threshold = 2.0;   // Default = 2.0 (insufficient for e.g., hill sequence)
//  sf_params.inlier_threshold = 2.7;   // May be required for the hill sequence
  sf_params.bucket.max_features = 15;    // Default = 2
  // VO is computed using the color frames.
  sf_params.calib.cu = left_color_proj(0, 2);
  sf_params.calib.cv = left_color_proj(1, 2);
  sf_params.calib.f  = left_color_proj(0, 0);

  auto sparse_sf_provider = new instreclib::VisoSparseSFProvider(sf_params);

  assert((FLAGS_dynamic_mode || !FLAGS_direct_refinement) && "Cannot use direct refinement in non-dynamic mode.");

  auto evaluation = new dynslam::eval::Evaluation(dataset_root,
                                                  *input_out,
                                                  velo_to_left_gray_cam,
                                                  left_color_proj,
                                                  right_color_proj,
                                                  baseline_m,
                                                  frame_size(0),  // width
                                                  frame_size(1),  // height
                                                  driver_settings->sceneParams.voxelSize,
                                                  FLAGS_direct_refinement,
                                                  FLAGS_dynamic_mode,
                                                  FLAGS_use_depth_weighting,
                                                  FLAGS_semantic_evaluation);

  Vector2i input_shape((*input_out)->GetRgbSize().width, (*input_out)->GetRgbSize().height);
  *dyn_slam_out = new DynSlam(
      driver,
      segmentation_provider,
      sparse_sf_provider,
      evaluation,
      input_shape,
      left_color_proj.cast<float>(),
      right_color_proj.cast<float>(),
      baseline_m,
      FLAGS_direct_refinement,
      FLAGS_dynamic_mode,
      FLAGS_fusion_every
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
    cerr << "The --dataset_root=<path> flag must be set." << endl;
    return -1;
  }

  dynslam::DynSlam *dyn_slam;
  dynslam::Input *input;
  BuildDynSlamKittiOdometry(dataset_root, &dyn_slam, &input);

  dynslam::gui::PangolinGui pango_gui(dyn_slam, input);
  pango_gui.Run();

  delete dyn_slam;
  delete input;
}

