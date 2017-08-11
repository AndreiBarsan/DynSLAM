
#include "Evaluation.h"
#include "ILidarEvalCallback.h"

namespace dynslam {
namespace eval {

void Evaluation::EvaluateFrame(Input *input, DynSlam *dyn_slam) {

  // TODO(andrei): Fused depth maps!

  if (dyn_slam->GetCurrentFrameNo() > 0) {
    cout << "Starting evaluation of current frame..." << endl;

    if (this->eval_tracklets_) {
      EvaluateTracking(input, dyn_slam);
    }

    DepthFrameEvaluation evals = EvaluateFrame(input->GetCurrentFrame() - 1, input, dyn_slam);
    csv_depth_dump_.Write(evals);

    // ...and print some quick info in real time as well.
    cout << "Evaluation complete:" << endl;
    bool missing_depths_are_errors = false;
    if(missing_depths_are_errors) {
      cout << "(Missing = errors)" << endl;
    }
    else {
      cout << "(Not counting missing as errors.)" << endl;
    }
    for (auto &eval : evals.evaluations) {
      cout << "Evaluation on frame #" << evals.meta.frame_idx << ", delta max = "
           << setw(2) << eval.delta_max
           << "   Fusion accuracy = " << setw(7) << setprecision(3)
           << eval.fused_result.GetCorrectPixelRatio(missing_depths_are_errors)
           << " @ " << eval.fused_result.correct_count << " correct px "
           << " | Input accuracy  = " <<  setw(7) << setprecision(3)
           << eval.input_result.GetCorrectPixelRatio(missing_depths_are_errors)
          << " @ " << eval.input_result.correct_count << " correct px "
           << endl;
    }
  }
}

DepthFrameEvaluation Evaluation::EvaluateFrame(int frame_idx,
                                               Input *input,
                                               DynSlam *dyn_slam
) {
  auto lidar_pointcloud = velodyne_->ReadFrame(frame_idx);

  // TODO(andrei): Pose history in dynslam; will be necessary in delayed evaluation, as well
  // as if you wanna preview cute little frustums!
  if (frame_idx != input->GetCurrentFrame() - 1) {
    throw runtime_error("Cannot yet access old poses for evaluation.");
  }
  Eigen::Matrix4f epose = dyn_slam->GetPose().inverse();
  auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

  const float *rendered_depthmap = dyn_slam->GetStaticMapRaycastDepthPreview(pango_pose);
  auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
  auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
  input->GetFrameCvImages(frame_idx, input_rgb, input_depthmap);

  int width = dyn_slam->GetInputWidth();
  int height = dyn_slam->GetInputHeight();
  float min_depth_meters = input->GetDepthProvider()->GetMinDepthMeters();
  float max_depth_meters = input->GetDepthProvider()->GetMaxDepthMeters();

  // TODO(andrei): Once you get this working, wrap images in OpenCV objects for readability and flexibility.
  std::vector<DepthEvaluation> evals;
  const int kLargestMaxDelta = 12;
  bool compare_on_intersection = true;
  for(uint delta_max = 1; delta_max <= kLargestMaxDelta; ++delta_max) {
    evals.push_back(EvaluateDepth(lidar_pointcloud,
                                  rendered_depthmap,
                                  *input_depthmap,
                                  velodyne_->velodyne_to_rgb,
                                  dyn_slam->GetLeftRgbProjectionMatrix().cast<double>(),
                                  dyn_slam->GetRightRgbProjectionMatrix().cast<double>(),
                                  dyn_slam->GetStereoBaseline(),
                                  width,
                                  height,
                                  min_depth_meters,
                                  max_depth_meters,
                                  delta_max,
                                  compare_on_intersection,
                                  nullptr));
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());
  return DepthFrameEvaluation(std::move(meta), max_depth_meters, std::move(evals));
}

DepthEvaluation Evaluation::EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                          const float *const rendered_depth,
                                          const cv::Mat1s &input_depth_mm,
                                          const Eigen::Matrix4d &velo_to_left_gray_cam,
                                          const Eigen::Matrix34d &proj_left_color,
                                          const Eigen::Matrix34d &proj_right_color,
                                          const float baseline_m,
                                          const int frame_width,
                                          const int frame_height,
                                          const float min_depth_meters,
                                          const float max_depth_meters,
                                          const uint delta_max,
                                          const bool compare_on_intersection,
                                          ILidarEvalCallback *callback) const {
  const float left_focal_length_px = static_cast<float>(proj_left_color(0, 0));
  long missing_rendered = 0;
  long errors_rendered = 0;
  long correct_rendered = 0;
  long missing_input = 0;
  long errors_input = 0;
  long correct_input = 0;
  long measurements = 0;

  int epi_errors = 0;

  for (int i = 0; i < lidar_points.rows(); ++i) {
    Eigen::Vector4d velo_point = lidar_points.row(i).cast<double>();
    // Ignore the reflectance; we only care about 3D homogeneous coordinates.
    velo_point(3) = 1.0f;
    Eigen::Vector4d cam_point = velo_to_left_gray_cam * velo_point;
    cam_point /= cam_point(3);

    double velo_z = cam_point(2);

    if (velo_z < min_depth_meters || velo_z > max_depth_meters) {
      continue;
    }

    Eigen::Vector3d velo_2d_left = proj_left_color * cam_point;
    Eigen::Vector3d velo_2d_right = proj_right_color * cam_point;
    velo_2d_left /= velo_2d_left(2);
    velo_2d_right /= velo_2d_right(2);

    int row_left = static_cast<int>(round(velo_2d_left(1)));
    int col_left = static_cast<int>(round(velo_2d_left(0)));
    int row_right = static_cast<int>(round(velo_2d_right(1)));
//    int col_right = static_cast<int>(round(velo_2d_right(0)));

    // We ignore LIDAR points which fall outside the left camera's retina.
    if (col_left < 0 || col_left >= frame_width ||
        row_left < 0 || row_left >= frame_height) {
      continue;
    }

    if (row_left != row_right) {
      int delta = row_left - row_right;
      float fdelta = velo_2d_left(1) - velo_2d_right(1);

      // Note that the LIDAR pointclouds aren't perfectly aligned, so this could occasionally happen.
      // In particular, this can happen when the car passes large trucks very closely, it seems.
      if (abs(fdelta) > 1.2) {
        epi_errors++;
        cerr << "Warning: epipolar violation! Delta: " << delta << "; float: "
             << fdelta << endl;
        cerr << "Whereby left float row was: " << velo_2d_left(1) << " and right float row was: "
             << velo_2d_right(1) << endl;
      }
    }

    const float lidar_disp = velo_2d_left(0) - velo_2d_right(0);
    if (lidar_disp < 0.0f) {
      throw std::runtime_error("Negative disparity in ground truth.");
    }

    int idx_in_rendered = row_left * frame_width + col_left;
    const float rendered_depth_m = rendered_depth[idx_in_rendered];
    const float input_depth_m = input_depth_mm.at<short>(row_left, col_left) / 1000.0f;
    if (input_depth_mm.at<short>(row_left, col_left) < 0) {
      cerr << "WARNING: Input depth negative value of " << input_depth_mm.at<short>(row_left, col_left)
           << " at "  << row_left << ", " << col_left << "." << endl;
    }

    // Old code which rounded before computing the disparity; slightly less accurate. We don't ever
    // have to round, actually!
//    const int rendered_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / rendered_depth_m));
//    const uint rendered_disp_delta = static_cast<uint>(std::abs(rendered_disp - lidar_disp));
//    const int input_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / input_depth_val));
//    const uint input_disp_delta = static_cast<uint>(std::abs(input_disp - lidar_disp));

    // Units of measurement: px = (m * px) / m;
    const float rendered_disp = baseline_m * left_focal_length_px / rendered_depth_m;
    const float rendered_disp_delta = fabs(rendered_disp - lidar_disp);
    const float input_disp = baseline_m * left_focal_length_px / input_depth_m;
    const float input_disp_delta = fabs(input_disp - lidar_disp);

    /// We want to compare the fusion and the input map only where they're both present, since
    /// otherwise the fusion covers a much larger area, so its evaluation is tougher.
    // experimental metric
    if (compare_on_intersection && (fabs(input_depth_m) < 1e-5 || fabs(rendered_depth_m) < 1e-5)) {
      missing_input++;
      missing_rendered++;
    }
    else {
      if (fabs(input_depth_m) < 1e-5) {
        missing_input++;
      } else {
//        if (input_disp_delta > delta_max && (input_disp_delta > 0.05 * lidar_disp)) {
        if (input_disp_delta > delta_max) {
          errors_input++;
        } else {
          correct_input++;
        }
      }

      if (rendered_depth_m < 1e-5) {
        missing_rendered++;
      } else {
//        if (rendered_disp_delta > delta_max && (rendered_disp_delta > 0.05 * lidar_disp)) {
        if (rendered_disp_delta > delta_max) {
          errors_rendered++;
        } else {
          correct_rendered++;
        }
      }
    }

    measurements++;

    if (nullptr != callback) {
      callback->LidarPoint(i,
                           velo_2d_left,
                           rendered_disp,
                           rendered_depth_m,
                           input_disp,
                           input_depth_m,
                           lidar_disp,
                           frame_width,
                           frame_height);
    }

  }

  if (epi_errors > 5) {
    cerr << "WARNING: Found " << epi_errors << " possible epipolar violations in the ground truth, "
         << "out of " << measurements << "." << endl;
  }

  DepthResult rendered_result(measurements, errors_rendered, missing_rendered, correct_rendered);
  DepthResult input_result(measurements, errors_input, missing_input, correct_input);

  return DepthEvaluation(delta_max,
                         std::move(rendered_result),
                         std::move(input_result));
}



// Track = ours, tracklet = ground truth
int GetBestOverlapping(
    const vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>> candidates,
    const TrackFrame &frame
) {
  using namespace instreclib::utils;

  // TODO(andrei): render the pose est. error(s) on the segmentation preview, if available, like
  // next to the [D] labels, for instance.
  int best_overlap = -1;
  int best_idx = -1;

  // Convention: (0, 0) is top-left.
  auto our_box = frame.instance_view.GetInstanceDetection().conservative_mask->GetBoundingBox();

  // Minimum overlap that we care about is 30x30 pixels.
  const int kMinOverlap = 30 * 30;
  for(int i = 0; i < candidates.size(); ++i){
    BoundingBox theirs = BoundingBox::RoundCoords(candidates[i].bbox_2d);
    int overlap = theirs.IntersectWith(our_box).GetArea();

    cout << "Candidate [" << candidates[i].track_id << "] overlap: " << overlap << "." << endl;
    if(overlap > best_overlap && overlap > kMinOverlap) {
      cout << "Best overlap updated." << endl;
      best_overlap = overlap;
      best_idx = i;
    }
  }

  return best_idx;
}

// Computes the relative pose between the two tracklet frames
Eigen::Matrix4d GetRelativeGTPose(const TrackletFrame &first, const TrackletFrame &second) {
  assert(first.track_id == second.track_id && "Must compute relative pose between frames from the "
      "same ground truth track.");

  double angle_delta = second.rotation_y - first.rotation_y;

  Eigen::Matrix3d rot;
  rot << cos(angle_delta),  0, sin(angle_delta),
                        0,  1,                0,
         -sin(angle_delta), 0, cos(angle_delta);

  // Note: the location is always the middle of the bbox, so this should be accurate even if the
  // bbox dimensions vary.
  Eigen::Vector3d trans = second.location_cam_m - first.location_cam_m;

  Eigen::Matrix4d transform;
  transform.block(0, 0, 3, 3) = rot;
  transform.block(0, 3, 3, 1) = trans;
  transform(3, 3) = 1.0;

  return transform;
}

void Evaluation::EvaluateTracking(Input *input, DynSlam *dyn_slam) {
  int cur_input_frame = input->GetCurrentFrame();
  int cur_time = dyn_slam->GetCurrentFrameNo();

  // Need both since otherwise we couldn't compute GT.
  if(frame_to_tracklets_.cend() == frame_to_tracklets_.find(cur_input_frame) &&
     frame_to_tracklets_.cend() == frame_to_tracklets_.find(cur_input_frame - 1)
   ) {
    cout << "No GT tracklets for frame [" << cur_input_frame << "] and its predecessor." << endl;
    return;
  }

  // Note: the KITTI benchmark, despite proving 3D GT, only evaluates 2D bounding box performance,
  // so there seems to be no de facto standard for evaluating 3D pose estimation performances.
  // "We evaluate 2D 0-based bounding boxes in each image."
  cout << "[Tracklets] Frame [" << cur_input_frame << "] from the input sequence has some data!" << endl;
  auto current_gt = frame_to_tracklets_[cur_input_frame];
  auto prev_gt = frame_to_tracklets_[cur_input_frame - 1];

  InstanceTracker &instance_tracker = dyn_slam->GetInstanceReconstructor()->GetInstanceTracker();
  for (auto &pair : instance_tracker.GetActiveTracks()) {
    const Track &track = pair.second;
    if (track.GetState() == TrackState::kDynamic && track.GetLastFrame().frame_idx == cur_time-1) {
      const TrackFrame &latest_frame = track.GetLastFrame();

      // This is not at all ideal, but should work for a coarse evaluation.
      int best_overlap_id = GetBestOverlapping(current_gt, latest_frame);

      if (best_overlap_id >= 0) {
        int current_gt_tid = current_gt[best_overlap_id].track_id;
        cout << "[Tracklets] Found a good GT for track " << track.GetId() << " (GT track ID "
             << current_gt_tid << ")." << endl;

        // Find in previous frame and compute the relative pose for comparing.
        for(auto &dude : prev_gt) {
          if (dude.track_id == current_gt_tid) {
            Eigen::Matrix4d rel = GetRelativeGTPose(dude, current_gt[best_overlap_id]);
            cout << "current z-angle (deg): " << current_gt[best_overlap_id].rotation_y * 180/M_PI << endl;
            cout << "Relative transform computed:" << endl << rel << endl << endl;

            // TODO(andrei): Think about this in detail: when you're doing the reconstruction, the
            // egomotion compensation you're doing basically brings them into the car's coordinate
            // frame, right?

            Eigen::Matrix4d ego = dyn_slam->GetLastEgomotion().cast<double>();
//            Eigen::Matrix4d result = ego * rel;

//            cout << "Ego-compensated: " << endl << result << endl << endl;
            cout << "Egomotion is:" << endl << ego << endl << endl;

            const Eigen::Matrix4d &computed_rel_pose = latest_frame.relative_pose->Get().matrix_form;
            cout << "And the computed internal relative pose: " << endl << computed_rel_pose
                 << endl << endl;

            Eigen::Matrix4f delta = (computed_rel_pose.inverse() * rel).cast<float>();
            cout << "Delta: " << endl << delta << endl << endl;
            float te = utils::TranslationError(delta);
            float re = utils::RotationError(delta);

            cout << "Translation error: " << te << "| Rotation error: " << re << endl;
          }
        }
      }
      else {
        cout << "[Tracklets] No GT match for track #" << track.GetId() << "." << endl;
      }
    }
  }

  // for every active track labeled as dynamic:
  //    use 2D bbox matching to find corresponding GT track
  //    compute most recent GT motion in the camera frame (default)
  //    compute most recent track motion in the camera frame (independent motion with egomotion removed)
  //    compute rotational and translational errors
}

}
}
