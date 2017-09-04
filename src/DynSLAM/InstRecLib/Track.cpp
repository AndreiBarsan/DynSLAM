

#include "Track.h"
#include "InstanceTracker.h"
#include "../../libviso2/src/viso.h"

namespace instreclib {
namespace reconstruction {

using namespace std;
using namespace dynslam::utils;
using namespace instreclib::segmentation;
using namespace instreclib::utils;

float Track::ScoreMatch(const TrackFrame& new_frame) const {
  // TODO(andrei): Use fine mask, not just bounding box.
  // TODO(andrei): Ensure this is modular enough to allow many different matching strategies.
  // TODO-LOW(andrei): Take time into account---if I overlap perfectly but with a very old track,
  // the score should probably be discounted.

  assert(!this->frames_.empty() && "A track with no frames cannot exist.");
  const TrackFrame& latest_frame = this->frames_[this->frames_.size() - 1];

  // We don't want to accidentally add multiple segments from the same frame to the same track.
  // This is not 100% elegant, but it works.
  int delta_time = new_frame.frame_idx - this->GetEndTime();

  if (delta_time == 0) {
    return 0.0f;
  }

  const InstanceDetection& new_detection = new_frame.instance_view.GetInstanceDetection();
  const InstanceDetection& latest_detection = latest_frame.instance_view.GetInstanceDetection();

  // We don't want to associate segments from different classes.
  // TODO(andrei): Sometimes the segmentation pipeline may flicker between, e.g., ``car'' and
  // ``truck'' so we may want a more complex reasoning system here in the future.
  if (new_detection.class_id != latest_detection.class_id) {
    return 0.0f;
  }

  const BoundingBox& new_bbox = new_detection.GetCopyBoundingBox();
  const BoundingBox& last_bbox = latest_detection.GetCopyBoundingBox();

  // Score the overlap using the standard intersection-over-union (IoU) measure.
  int intersection = last_bbox.IntersectWith(new_bbox).GetArea();
  int union_area = new_bbox.GetArea() + last_bbox.GetArea() - intersection;
  float area_score = static_cast<float>(intersection) / union_area;

  // Modulate the score by the detection probability. If we see a good overlap but it's a dodgy
  // detection, we may not want to add it to the track. For instance, when using MNC for
  // segmentation, it may sometimes detect both part of a car and the entire car as separate
  // instances. Luckily, in these situations, the proper full detection gets a score near 1.0,
  // while the partial one is around 0.5-0.6. We'd prefer to fuse in the entire car, so we take
  // the probability into account. Similarly, in a future frame, we prefer adding the new data to
  // the track with the most confident detections.
  float score = area_score * new_detection.class_probability * latest_detection.class_probability;

  float time_discount = 1.0f;
  // 1 = no gap in the track
  if (delta_time == 2) {
    time_discount = 0.5f;
  }
  else if (delta_time > 2) {
    time_discount = 0.25;
  }

  return score * time_discount;
}

string Track::GetAsciiArt() const {
  stringstream out;
  out << "Object #" << setw(4) << id_ << " [";
  int idx = 0;
  for (const TrackFrame& frame : frames_) {
    while (idx < frame.frame_idx) {
      out << "   ";
      ++idx;
    }
    out << setw(3) << frame.frame_idx;
    ++idx;
  }
  out << "]";

  return out.str();
}

Option<Eigen::Matrix4d> Track::GetFramePose(size_t frame_idx) const {
  assert(frame_idx < GetFrames().size() && "Cannot get the relative pose of a non-existent frame.");

  // Skip the original very distant frames with no relative pose info.
  bool found_good_pose = false;
  Eigen::Matrix4d *pose = new Eigen::Matrix4d;
  pose->setIdentity();

  // Start from 1 since we care about relative pose to 1st frame.
  for (size_t i = 1; i <= frame_idx; ++i) {
    if (frames_[i].relative_pose->IsPresent()) {
      found_good_pose = true;

      const Eigen::Matrix4d &rel_pose = frames_[i].relative_pose->Get().matrix_form;
      *pose = rel_pose * (*pose);
    }
    else {
      // Gap caused by instances switching (static/dynamic) -> uncertain -> (static/dynamic).
      if (found_good_pose) {
        cout << "(static/dynamic) -> uncertain -> (static/dynamic) case detected; ignoring first "
            << "reconstruction attempt." << endl;
        found_good_pose = false;
        pose->setIdentity();
      }
    }
  }

  return Option<Eigen::Matrix4d>(pose);
}

dynslam::utils::Option<Eigen::Matrix4d> Track::GetFramePoseDeprecated(size_t frame_idx) const {
  assert(frame_idx < GetFrames().size() && "Cannot get the relative pose of a non-existent frame.");

  bool found_good_pose = false;
  Eigen::Matrix4d *pose = new Eigen::Matrix4d;
  pose->setIdentity();

  Eigen::Matrix4d first_good_cam_pose;
  first_good_cam_pose.setIdentity();

  for (size_t i = 1; i <= frame_idx; ++i) {
    if (frames_[i].relative_pose_world->IsPresent()) {
      if (! found_good_pose) {
        first_good_cam_pose = frames_[i].camera_pose.cast<double>();
        found_good_pose = true;
      }

      const Eigen::Matrix4d &rel_pose = frames_[i].relative_pose_world->Get().cast<double>();
      const Eigen::Matrix4d new_pose = rel_pose * (*pose);

      *pose = new_pose;
    }
    else {
      if (found_good_pose) {
        // This is OK even if the previos "streak" had triggered a reconstruction, since as soon
        // as we detect an resumed reconstruction after an interruption, we clear the reconstruction
        // volume.
        found_good_pose = false;
        pose->setIdentity();
      }
    }
  }

  if (track_state_ == TrackState::kStatic) {
    return dynslam::utils::Option<Eigen::Matrix4d>(new Eigen::Matrix4d(first_good_cam_pose));
  }

  if (found_good_pose) {
    Eigen::Matrix4d aux = first_good_cam_pose * *pose;
    *pose = aux;
    return dynslam::utils::Option<Eigen::Matrix4d>(pose);
  }
  else {
    return dynslam::utils::Option<Eigen::Matrix4d>::Empty();
  }
}

Option<Pose>* Track::EstimateInstanceMotion(
    const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &instance_raw_flow,
    const SparseSFProvider &ssf_provider,
    const vector<double> &initial_estimate
) {
  // This is a good conservative value, but we can definitely do better.
  // TODO(andrei): Try setting the minimum to 6-10, but threshold based on the final RMSE, flagging
  // pose estimates whose residual is above some value as invalid.
  // Note: 25 => OK results in most cases, but where cars are advancing from opposite direction
  //             in the hill sequence, this no longer works.
//  uint32_t kMinFlowVectorsForPoseEst = 25;
  uint32_t kMinFlowVectorsForPoseEst = 18;
  // technically 3 should be enough (because they're stereo-and-time 4-way correspondences, but
  // we're being a little paranoid).
  size_t flow_count = instance_raw_flow.size();

  if (instance_raw_flow.size() >= kMinFlowVectorsForPoseEst) {
    vector<double> instance_motion_delta = ssf_provider.ExtractMotion(instance_raw_flow, initial_estimate);
    if (instance_motion_delta.size() != 6) {
      // track information not available yet; idea: we could move this computation into the
      // track object, and use data from many more frames (if available).
      cerr << "Could not compute instance #" << GetId() << " delta motion from " << flow_count << " matches." << endl;
      return new Option<Pose>;
    } else {
      cout << "Successfully estimated the relative instance pose from " << flow_count
           << " matches." << endl;
      // This is a libviso2 matrix.
      Matrix delta_mx = VisualOdometry::transformationVectorToMatrix(instance_motion_delta);

      // We then invert it and convert it into an Eigen matrix.
      // TODO(andrei): Make this a utility.
      return new Option<Pose>(new Pose(
          instance_motion_delta,
          Eigen::Matrix4d((~delta_mx).val[0])
      ));
    }
  }
  else {
    cout << "Only " << flow_count << " scene flow points. Not estimating relative pose for track "
         << "#" << GetId() << "." << endl;
    return new Option<Pose>();
  }
}


void Track::Update(const Eigen::Matrix4f &egomotion,
                   const instreclib::SparseSFProvider &ssf_provider,
                   bool verbose) {

  long prev_frame_idx = static_cast<long>(frames_.size()) - 2;
  vector<double> initial_estimate = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  if (prev_frame_idx >= 0) {
    auto prev_pose = frames_[prev_frame_idx].relative_pose;
    if (prev_pose->IsPresent()) {
      // Perform warm start is previous relative pose is known.
      initial_estimate = prev_pose->Get().se3_form;
    }
  }

  // Vehicle motion INCLUDING camera egomotion.
  Option<Pose> *motion_delta = EstimateInstanceMotion(
      GetLastFrame().instance_view.GetFlow(),
      ssf_provider,
      initial_estimate
  );
  GetLastFrame().relative_pose = motion_delta;
  if (motion_delta->IsPresent()) {
    GetLastFrame().relative_pose_world = new Option<Eigen::Matrix4f>(new Eigen::Matrix4f(
        egomotion * motion_delta->Get().matrix_form.cast<float>()));
  }
  else {
    GetLastFrame().relative_pose_world = new Option<Eigen::Matrix4f>();
  }

  int current_frame_idx = GetLastFrame().frame_idx;

  // TODO(andrei): Buffer region for rot/trans error within which the state of the track is still
  // left as uncertain.
  switch(track_state_) {
    case kUncertain:
      if (motion_delta->IsPresent()) {
        Eigen::Matrix4f error = egomotion * motion_delta->Get().matrix_form.cast<float>();
        float trans_error = TranslationError(error);
        float rot_error = RotationError(error);

        if (verbose) {
          cout << "Object " << id_ << " has " << setw(8) << setprecision(4) << trans_error
               << " translational error w.r.t. the egomotion." << endl;
          cout << "Rotation error: " << rot_error << "(currently unused)" << endl;
          cout << endl << "ME: " << endl << egomotion << endl;
          cout << endl << "Object: " << endl << motion_delta->Get().matrix_form << endl;
        }

        if (trans_error > kTransErrorThresholdHigh) {
          if (verbose) {
            cout << id_ << ": Uncertain -> Dynamic object!" << endl;
          }
          this->track_state_ = kDynamic;
        }
        else if (trans_error < kTransErrorThresholdLow) {
          if (verbose) {
            cout << id_ << ": Uncertain -> Static object!" << endl;
          }
          // If the motion is below the threshold, meaning that the object is stationary, set it to
          // identity to make the result more accurate.
          motion_delta->Get().SetIdentity();
          this->track_state_ = kStatic;
        }
        else {
          if (verbose) {
            cout << id_ << ": Uncertain -> Still uncertain because of ambiguous motion!" << endl;
          }
        }

        this->last_known_motion_ = motion_delta->Get();
        this->last_known_motion_world_ = egomotion * motion_delta->Get().matrix_form.cast<float>();
        this->last_known_motion_time_ = current_frame_idx;
      }

      if (track_state_ != kUncertain) {
        // We just switched states

        if (HasReconstruction()) {
          // Corner case: an instance which was static or dynamic, started being reconstructed, then
          // became uncertain again, and then was labeled as static or dynamic once again. In this
          // case, we have no way of registering our new measurements to the existing
          // reconstruction, so we discard it in order to start fresh.
          cout << "Uncertain -> Static/Dynamic BUT a reconstruction was already present. "
               << "Resetting reconstruction to avoid corruption." << endl;
          reconstruction_->Reset();
        }
      }

      break;

    case kStatic:
    case kDynamic:
      assert(last_known_motion_time_ >= 0);

      int frameThreshold = (track_state_ == kStatic) ? kMaxUncertainFramesStatic :
                           kMaxUncertainFramesDynamic;

      if (motion_delta->IsPresent()) {
        if (track_state_ == kStatic) {
          this->last_known_motion_.SetIdentity();
          this->last_known_motion_world_.setIdentity();

          GetLastFrame().relative_pose_world->Get().setIdentity();
        }
        else {
          this->last_known_motion_ = motion_delta->Get();
          this->last_known_motion_world_ = motion_delta->Get().matrix_form.cast<float>();
        }

        this->last_known_motion_time_ = current_frame_idx;
      }
      else {
        int motion_age = current_frame_idx - last_known_motion_time_;
        if (motion_age > frameThreshold) {
          if (verbose) {
            cout << id_ << ": " << GetStateLabel() << " -> Uncertain because the relative motion "
                 << "could not be evaluated over the last " << frameThreshold << " frames."
                 << endl;
          }

          this->track_state_ = kUncertain;
        }
        else {
          // Assume constant motion for small gaps in the track.
          GetLastFrame().relative_pose = new Option<Pose>(new Pose(last_known_motion_));
          GetLastFrame().relative_pose_world = new Option<Eigen::Matrix4f>(new Eigen::Matrix4f(last_known_motion_world_));
        }
      }
      break;
  }
}

}  // namespace reconstruction
}  // namespace instreclib
