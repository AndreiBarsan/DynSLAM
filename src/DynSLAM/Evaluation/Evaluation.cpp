
#include "Evaluation.h"
#include "ILidarEvalCallback.h"
#include "EvaluationCallback.h"
#include "SegmentedEvaluationCallback.h"

namespace dynslam {
namespace eval {

void PrettyPrintStats(const string &label, const DepthFrameEvaluation &evals) {
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
    cout << "[" << label << "] Evaluation on frame #" << evals.meta.frame_idx << ", delta max = "
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

void Evaluation::EvaluateFrame(Input *input,
                               DynSlam *dyn_slam,
                               int frame_idx,
                               bool enable_compositing
) {
  if (frame_idx < 0) {
    cerr << "Cannot evaluate negative frame [" << frame_idx << "]." << endl;
    return;
  }

  cout << "Starting evaluation of frame [" << frame_idx << "].." << endl;

  if (this->eval_tracklets_) {
    vector<TrackletEvaluation> tracklet_evals = EvaluateTracking(input, dyn_slam);

    for (const TrackletEvaluation &eval : tracklet_evals) {
      csv_tracking_dump_.Write(eval);
    }
  }

  if (!velodyne_->FrameAvailable(frame_idx)) {
    cerr << "WARNING: Skipping evaluation for frame #" << frame_idx << " since no "
         << "ground truth is available for it (normal for very short chunks)." << endl;

    return;
  }

  if (separate_static_and_dynamic_) {
    cout << "Evaluation of frame [" << frame_idx << "] will compute separate stats for static "
         << "and dynamic elements of the scene." << endl;
    auto static_dynamic = EvaluateFrameSeparate(frame_idx, enable_compositing, input, dyn_slam);
    auto static_evals = static_dynamic.first;
    auto dynamic_evals = static_dynamic.second;

    csv_static_depth_dump_.Write(static_evals);
    csv_dynamic_depth_dump_.Write(dynamic_evals);

    PrettyPrintStats("Static Map", static_evals);
    PrettyPrintStats("Dynamic", dynamic_evals);
  } else {
    cout << "Evaluation of frame [" << frame_idx << "] will compute unified stats for both "
         << "static and dynamic parts of the scene." << endl;

    DepthFrameEvaluation evals = EvaluateFrame(frame_idx, enable_compositing, input, dyn_slam);
    csv_unified_depth_dump_.Write(evals);

    PrettyPrintStats("Unified", evals);
  }
}

// TODO(andrei): Deduplicate copypasta code.
std::pair<DepthFrameEvaluation,
          DepthFrameEvaluation> Evaluation::EvaluateFrameSeparate(int dynslam_frame_idx,
                                                                  bool enable_compositing,
                                                                  Input *input,
                                                                  DynSlam *dyn_slam) {
  int input_frame_idx = input->GetFrameOffset() + dynslam_frame_idx;
  auto lidar_pointcloud = velodyne_->ReadFrame(input_frame_idx);
  int pose_idx = dynslam_frame_idx + 1;
  Eigen::Matrix4f epose = dyn_slam->GetPoseHistory()[pose_idx];
  cout << "Getting DynSLAM pose[" << pose_idx << "] from a total history of length "
       << dyn_slam->GetPoseHistory().size() << endl;
  cout << "This corresponds to input frame " << input_frame_idx << endl;

  auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

  const float *rendered_depthmap = dyn_slam->GetStaticMapRaycastDepthPreview(pango_pose, enable_compositing);
  auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
  auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
  input->GetFrameCvImages(input_frame_idx, input_rgb, input_depthmap);

  const int kLargestMaxDelta = 12;
  bool compare_on_intersection = true;
  const bool kKittiStyle = true;
  const bool kNonKittiStyle = false;
  float kKittiDeltaMax = 3.0f;

  auto seg = dyn_slam->GetLatestSeg();
  auto reconstructor = dyn_slam->IsDynamicMode() ? dyn_slam->GetInstanceReconstructor() : nullptr;

  std::vector<ILidarEvalCallback *> callbacks;
  callbacks.push_back(new SegmentedEvaluationCallback(0.5f, compare_on_intersection,
                                                      kNonKittiStyle, seg.get(), reconstructor));

  for (int delta_max = 1; delta_max <= kLargestMaxDelta; ++delta_max) {
    callbacks.push_back(new SegmentedEvaluationCallback(delta_max,
                                               compare_on_intersection,
                                               kNonKittiStyle, seg.get(), reconstructor));
  }

  // Finally, perform the KITTI-style depth evaluation.
  callbacks.push_back(new SegmentedEvaluationCallback(kKittiDeltaMax,
                                             compare_on_intersection,
                                             kKittiStyle, seg.get(), reconstructor));

  EvaluateDepth(lidar_pointcloud, rendered_depthmap, *input_depthmap, callbacks);

  std::vector<DepthEvaluation> static_evals;
  std::vector<DepthEvaluation> dynamic_evals;
  // Forced dynamic casts because std::vector is not covariant, grumble, grumble...
  for(ILidarEvalCallback *callback : callbacks) {
    auto *eval_callback = dynamic_cast<SegmentedEvaluationCallback *>(callback);
    assert(nullptr != eval_callback && "Only evaluation callbacks are supported at this point.");

    static_evals.push_back(std::move(eval_callback->GetStaticEvaluation()));
    dynamic_evals.push_back(std::move(eval_callback->GetDynamicEvaluation()));
    delete callback;
  }

  DepthEvaluationMeta meta(dynslam_frame_idx, input->GetDatasetIdentifier());
  return make_pair<DepthFrameEvaluation, DepthFrameEvaluation>(
      DepthFrameEvaluation(meta, max_depth_m_, std::move(static_evals)),
      DepthFrameEvaluation(meta, max_depth_m_, std::move(dynamic_evals)));
}

DepthFrameEvaluation Evaluation::EvaluateFrame(int frame_idx,
                                               bool enable_compositing,
                                               Input *input,
                                               DynSlam *dyn_slam) {
  throw std::runtime_error("Not supported at the moment.");
  auto lidar_pointcloud = velodyne_->ReadFrame(frame_idx);

  if (frame_idx != input->GetCurrentFrame() - 1) {
    throw runtime_error("Cannot yet access old poses for evaluation.");
  }
  Eigen::Matrix4f epose = dyn_slam->GetPose().inverse();
  auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

  const float *rendered_depthmap = dyn_slam->GetStaticMapRaycastDepthPreview(pango_pose, enable_compositing);
  auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
  auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
  input->GetFrameCvImages(frame_idx, input_rgb, input_depthmap);

  const int kLargestMaxDelta = 12;
  bool compare_on_intersection = true;

  // Mini hack for readability. Used to select between KITTI-style and non-KITTI style evaluation,
  // whereby the former also needs a disparity error to be >5% GT disp, in addition to >=delta_max,
  // for it to count as inaccurate.
  const bool kKittiStyle = true;
  const bool kNonKittiStyle = false;
  float kKittiDeltaMax = 3.0f;

  // TODO(andrei): Pass evaluation callbacks, one for every configuration, and the gather their
  // DepthEvaluation objects.

  std::vector<ILidarEvalCallback *> callbacks;

  // Also eval with max disparity of 0.5px, in the spirit of middlebury, even though this level is
  // susceptible to noise. (Just in case!)
  callbacks.push_back(new EvaluationCallback(0.5f, compare_on_intersection, kNonKittiStyle));

  for (int delta_max = 1; delta_max <= kLargestMaxDelta; ++delta_max) {
    callbacks.push_back(new EvaluationCallback(delta_max,
                                               compare_on_intersection,
                                               kNonKittiStyle));
  }

  // Finally, perform the KITTI-style depth evaluation.
  callbacks.push_back(new EvaluationCallback(kKittiDeltaMax,
                                             compare_on_intersection,
                                             kKittiStyle));

  EvaluateDepth(lidar_pointcloud, rendered_depthmap, *input_depthmap, callbacks);

  std::vector<DepthEvaluation> evals;
  // Forced dynamic casts because std::vector is not covariant, grumble, grumble...
  for(ILidarEvalCallback *callback : callbacks) {
    auto *eval_callback = dynamic_cast<EvaluationCallback *>(callback);
    assert(nullptr != eval_callback && "Only evaluation callbacks are supported at this point.");

    evals.push_back(std::move(eval_callback->GetEvaluation()));
    delete callback;
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());
  return DepthFrameEvaluation(meta, max_depth_m_, std::move(evals));
}


/// \return Whether the reading is valid and should be processed further.
bool Evaluation::ProjectLidar(const Eigen::Vector4f &velodyne_reading,
                              Eigen::Vector3d& out_velo_2d_left,
                              Eigen::Vector3d& out_velo_2d_right
) const {
  Eigen::Vector4d velo_point = velodyne_reading.cast<double>();

  // Ignore the reflectance; we only care about 3D homogeneous coordinates.
  velo_point(3) = 1.0f;
  Eigen::Vector4d cam_point = velo_to_left_gray_cam_ * velo_point;
  cam_point /= cam_point(3);

  double velo_z = cam_point(2);

  if (velo_z < min_depth_m_ || velo_z > max_depth_m_) {
    return false;
  }

  out_velo_2d_left = proj_left_color_ * cam_point;
  out_velo_2d_right = proj_right_color_ * cam_point;
  out_velo_2d_left /= out_velo_2d_left(2);
  out_velo_2d_right /= out_velo_2d_right(2);

  return true;
}


void Evaluation::EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                          const float *const rendered_depth,
                                          const cv::Mat1s &input_depth_mm,
                                          const std::vector<ILidarEvalCallback *> &callbacks) const {
  int valid_lidar_points = 0;
  int epi_errors = 0;
  for (int i = 0; i < lidar_points.rows(); ++i) {
    Eigen::Vector3d velo_2d_left, velo_2d_right;
    if (! ProjectLidar(lidar_points.row(i), velo_2d_left, velo_2d_right)) {
      continue;
    }

    int row_left = static_cast<int>(round(velo_2d_left(1)));
    int col_left = static_cast<int>(round(velo_2d_left(0)));
    int row_right = static_cast<int>(round(velo_2d_right(1)));
    if (col_left < 0 || col_left >= frame_width_ ||
        row_left < 0 || row_left >= frame_height_) {
      // We ignore LIDAR points which fall outside the left camera's frame.
      continue;
    }

    if (row_left != row_right) {
      float fdelta = velo_2d_left(1) - velo_2d_right(1);

      // Note that the LIDAR pointclouds aren't perfectly aligned, so this could occasionally happen.
      // In particular, this can happen when the car passes large trucks very closely, it seems.
      if (abs(fdelta) > 1.2) {
        epi_errors++;
      }
    }

    const float lidar_disp = velo_2d_left(0) - velo_2d_right(0);
    if (lidar_disp < 0.0f) {
      throw std::runtime_error("Negative disparity in ground truth.");
    }
    valid_lidar_points++;

    int idx_in_rendered = row_left * frame_width_ + col_left;
    const float rendered_depth_m = rendered_depth[idx_in_rendered];
    const float input_depth_m = input_depth_mm.at<short>(row_left, col_left) / 1000.0f;
    assert(input_depth_mm.at<short>(row_left, col_left) >= 0 && "Negative depth found in input.");

    // Units of measurement: px = (m * px) / m;
    const float rendered_disp = baseline_m_ * left_focal_length_px_ / rendered_depth_m;
    const float input_disp = baseline_m_ * left_focal_length_px_ / input_depth_m;

    for (ILidarEvalCallback *callback : callbacks) {
      callback->ProcessLidarPoint(i,
                                  velo_2d_left,
                                  rendered_disp,
                                  rendered_depth_m,
                                  input_disp,
                                  input_depth_m,
                                  lidar_disp,
                                  frame_width_,
                                  frame_height_);
    }
  }

  if (epi_errors > 5) {
    cerr << "WARNING: Found " << epi_errors << " possible epipolar violations in the ground truth, "
         << "out of " << valid_lidar_points << "valid LIDAR points." << endl;
  }
}

// Track = ours, tracklet = ground truth
int GetBestOverlapping(
    const vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>> &candidates,
    const TrackFrame &frame
) {
  using namespace instreclib::utils;
  int best_overlap = -1;
  int best_idx = -1;

  // Convention: (0, 0) is top-left.
  auto our_box = frame.instance_view.GetInstanceDetection().conservative_mask->GetBoundingBox();

  const int kMinOverlap = 30 * 30;
  for(int i = 0; i < static_cast<long>(candidates.size()); ++i){
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

vector<TrackletEvaluation> Evaluation::EvaluateTracking(Input *input, DynSlam *dyn_slam) {
  int cur_input_frame = input->GetCurrentFrame();
  int cur_time = dyn_slam->GetCurrentFrameNo();

  // Need both since otherwise we couldn't compute GT.
  if(frame_to_tracklets_.cend() == frame_to_tracklets_.find(cur_input_frame) &&
     frame_to_tracklets_.cend() == frame_to_tracklets_.find(cur_input_frame - 1)
   ) {
    cout << "No GT tracklets for frame [" << cur_input_frame << "] and its predecessor." << endl;
    return vector<TrackletEvaluation>();
  }

  vector<TrackletEvaluation> evaluations;

  // Note: the KITTI benchmark checks tracking accuracy in 2D, and evaluates the angular error of a
  // detection. No 3D box intersections are performed. The 3D object detection benchmark is
  // single-frame.
  cout << "[Tracklets] Frame [" << cur_input_frame << "] from the input sequence has some data!" << endl;
  auto current_gt = frame_to_tracklets_[cur_input_frame];
  auto prev_gt = frame_to_tracklets_[cur_input_frame - 1];

  InstanceTracker &instance_tracker = dyn_slam->GetInstanceReconstructor()->GetInstanceTracker();
  for (auto &pair : instance_tracker.GetActiveTracks()) {
    const Track &track = pair.second;
    if (track.GetState() == TrackState::kDynamic && track.GetLastFrame().frame_idx == cur_time - 1) {
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
            float trans_error = utils::TranslationError(delta);
            float rot_error = utils::RotationError(delta);

            cout << "Translation error: " << trans_error << "| Rotation error: " << rot_error << endl;

            evaluations.push_back(
                TrackletEvaluation(cur_input_frame, current_gt_tid, trans_error, rot_error));
          }
        }
      }
      else {
        cout << "[Tracklets] No GT match for track #" << track.GetId() << "." << endl;
      }
    }
  }

  return evaluations;
}

}
}
