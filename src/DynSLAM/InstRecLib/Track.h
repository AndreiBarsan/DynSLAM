#ifndef INSTRECLIB_TRACK_H
#define INSTRECLIB_TRACK_H

#include <Eigen/StdVector>
#include "InstanceSegmentationResult.h"
#include "InstanceView.h"
#include "../InfiniTamDriver.h"
#include "../Utils.h"
#include "../Defines.h"

namespace instreclib {
namespace reconstruction {

// Very naive holder of a SE3 transform + its matrix form.
// TODO(andrei): If you end up using this in the long run, make it nicer, like ITMPose.
struct Pose {
//  Eigen::Matrix<double, 6, 1> se3_form;
  std::vector<double> se3_form;
  Eigen::Matrix4d matrix_form;

  Pose() : se3_form({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}), matrix_form(Eigen::Matrix4d::Identity()) {}

  Pose(const vector<double> &se3_form, const Eigen::Matrix4d &matrix_form)
      : se3_form(se3_form), matrix_form(matrix_form) {}

  Pose(const Pose& other) {
    this->operator=(other);
  }

  Pose& operator=(const Pose& other) {
    se3_form = other.se3_form;
    matrix_form = other.matrix_form;
    return *this;
  }

  void SetIdentity() {
    matrix_form = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 6; ++i) {
      se3_form[i] = 0.0;
    }
  }

  SUPPORT_EIGEN_FIELDS;
};


/// \brief One frame of an instance track (InstRecLib::Reconstruction::Track).
struct TrackFrame {
  int frame_idx;
  InstanceView instance_view;

  /// \brief The camera pose at the time when this frame was observed.
  Eigen::Matrix4f camera_pose;

  /// \brief The relative pose to the previous frame in the track, if it could be computed.
  /// Includes the camera egomotion component, if present.
  dynslam::utils::Option<Pose> *relative_pose;

  // XXX: consider only storing this and using the camera egomotion history for aligning the frames
  // while reconstructing => correct compositing, as well as conceptually cleaner.
  /// \brief Relative pose to the previous frame, in world coordinates.
  dynslam::utils::Option<Eigen::Matrix4f> *relative_pose_world;

  TrackFrame(int frame_idx, const InstanceView& instance_view, const Eigen::Matrix4f &camera_pose)
      : frame_idx(frame_idx), instance_view(instance_view), camera_pose(camera_pose) {}

  SUPPORT_EIGEN_FIELDS;
};

/// \brief The states of an active track, which depend on the ability to estimate the relative pose
///        between subsequent frames. A track is uncertain until the relative motion between two
///        frames can be computed. It then switches to either static or dynamic depending on whether
///        the object's motion is significantly different from the camera's egomotion.
enum TrackState {
  kStatic,
  kDynamic,
  kUncertain
};

/// \brief A detected object's track through multiple frames.
/// Modeled as a series of detections, contained in the 'frames' field. Note that there can be
/// gaps in this list, due to frames where this particular object was not detected.
/// \todo In the long run, this class should be able to leverage a 3D reconstruction and something
///       like a Kalman Filter for motion tracking to predict an object's (e.g., car) pose in a
///       subsequent frame, in order to aid with tracking.
class Track {
 public:
  /// \brief The maximum number of frames with relative motion estimation failure before a static
  ///        object is reverted to the "Uncertain" state.
  int kMaxUncertainFramesStatic = 5;
  /// \see kMaxUncertainFramesStatic
  int kMaxUncertainFramesDynamic = 1;
  /// \brief Translation error threshold used to differentiate static from uncertain objects.
//  float kTransErrorThresholdLow =  0.033f;
// Note: 0.015 works better usually.
  float kTransErrorThresholdLow =  0.030f;
  /// \brief Translation error threshold used to differentiate dynamic from uncertain objects.
  float kTransErrorThresholdHigh = 0.550f;

  Track(int id) : id_(id),
                  reconstruction_(nullptr),
                  needs_cleanup_(false),
                  track_state_(TrackState::kUncertain)
  {}

  virtual ~Track() {
    if (reconstruction_.get() != nullptr) {
      fprintf(stderr, "Deleting track [%d] and its associated reconstruction!\n", id_);
    }
  }

  /// \brief Updates the track's state, and, if applicable, populates the most recent relative pose.
  void Update(const Eigen::Matrix4f &egomotion,
              const instreclib::SparseSFProvider &ssf_provider,
              bool verbose);

  /// \brief Evaluates how well this new frame would fit the existing track.
  /// \returns A goodness score between 0 and 1, where 0 means the new frame would not match this
  /// track at all, and 1 would be a perfect match.
  float ScoreMatch(const TrackFrame& new_frame) const;

  void AddFrame(const TrackFrame& new_frame) { frames_.push_back(new_frame); }

  size_t GetSize() const { return frames_.size(); }

  TrackFrame& GetLastFrame() { return frames_.back(); }

  const TrackFrame& GetLastFrame() const { return frames_.back(); }

  int GetStartTime() const { return frames_.front().frame_idx; }

  int GetEndTime() const { return frames_.back().frame_idx; }

  const std::vector<TrackFrame, Eigen::aligned_allocator<TrackFrame>>& GetFrames() const { return frames_; }

  const TrackFrame& GetFrame(int i) const { return frames_[i]; }
  TrackFrame& GetFrame(int i) { return frames_[i]; }

  int GetId() const { return id_; }

  std::string GetClassName() const {
    assert(frames_.size() > 0 || "Need at least one frame to determine a track's class.");
    return GetLastFrame().instance_view.GetInstanceDetection().GetClassName();
  }

  /// \brief Draws a visual representation of this feature track.
  /// \example For an object first seen in frame 11, then in frames 12, 13, and 16, this
  /// representation would look as follows:
  ///    [                                 11 12 13      16]
  std::string GetAsciiArt() const;

  bool HasReconstruction() const { return reconstruction_.get() != nullptr; }

  std::shared_ptr<dynslam::drivers::InfiniTamDriver>& GetReconstruction() {
    return reconstruction_;
  }

  const std::shared_ptr<dynslam::drivers::InfiniTamDriver>& GetReconstruction() const {
    return reconstruction_;
  }

  /// \brief Uses a series of goodness heuristics to establish whether the information contained in
  ///        this track's frames is good enough for a 3D reconstruction.
  bool EligibleForReconstruction() const {
    // Now that we have uncertain-dyn-static classification in place, this is not as important.
    return GetSize() >= 1;
  }

  /// \brief Returns the relative pose of the specified frame w.r.t. the first one.
  dynslam::utils::Option<Eigen::Matrix4d> GetFramePose(size_t frame_idx) const;

  /// \deprecated Used to return world pose, but that's no longer necessary. This now duplicates the
  /// functionality of 'GetFramePose'.
  /// TODO(andre): Safely remove this after the thesis deadline.
  dynslam::utils::Option<Eigen::Matrix4d> GetFramePoseDeprecated(size_t frame_idx) const;

  bool NeedsCleanup() const {
    return needs_cleanup_;
  }

  void SetNeedsCleanup(bool needs_cleanup) {
    this->needs_cleanup_ = needs_cleanup;
  }

  TrackState GetState() const {
    return track_state_;
  }

  string GetStateLabel() const {
    switch (track_state_) {
      case TrackState::kDynamic:
        return "Dynamic";
      case TrackState::kStatic:
        return "Static";
      case TrackState::kUncertain:
        return "Uncertain";
      default:
        throw runtime_error("Unsupported track type.");
    }
  }

  /// \brief Finds the first frame with a known relative pose, and returns the index of the frame
  ///        right before it.
  int GetFirstFusableFrameIndex() const {
    if (frames_.empty()) {
      return -1;
    }

    for (int i = 0; i < static_cast<int>(frames_.size()); ++i) {
      if (frames_[i].relative_pose->IsPresent()) {
        return max(0, i - 1);
      }
    }

    return -1;
  }

  void CountFusedFrame() {
    fused_frames_++;
  }

  void ReapReconstruction() {
    // TODO(andrei): Pass max fusion weight here and compute this in a smarter way.
    float factor = 0.33;
    int max_weight = 3;
    int reap_weight = max(1, min(max_weight, static_cast<int>(factor * fused_frames_)));
    cout << "Reaping track with max weight [" << reap_weight << "]." << endl;
    reconstruction_->Reap(reap_weight);
  }

 private:
  /// \brief A unique identifier for this particular track.
  int id_;
  std::vector<TrackFrame, Eigen::aligned_allocator<TrackFrame>> frames_;

  /// \brief A pointer to a 3D reconstruction of the object in this track.
  /// Is set to `nullptr` if no reconstruction is available.
  std::shared_ptr<dynslam::drivers::InfiniTamDriver> reconstruction_;

  /// \brief Whether the reconstruction is pending a full voxel decay iteration.
  bool needs_cleanup_;

  TrackState track_state_;

  // Used for the constant velocity assumption in tracking.
  int last_known_motion_time_ = -1;
  Pose last_known_motion_;
  Eigen::Matrix4f last_known_motion_world_;

  /// \brief The number of frames fused in the reconstruction.
  int fused_frames_ = 0;


  dynslam::utils::Option<Pose>* EstimateInstanceMotion(
      const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &instance_raw_flow,
      const SparseSFProvider &ssf_provider,
      const vector<double> &initial_estimate);

  SUPPORT_EIGEN_FIELDS;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_TRACK_H
