

#ifndef INSTRECLIB_TRACK_H
#define INSTRECLIB_TRACK_H

#include <vector>
#include "InstanceSegmentationResult.h"
#include "InstanceView.h"
#include "../InfiniTamDriver.h"
#include "../Utils.h"
#include "../Defines.h"

namespace instreclib {
namespace reconstruction {

/// \brief One frame of an instance track (InstRecLib::Reconstruction::Track).
struct TrackFrame {
  int frame_idx;
  InstanceView instance_view;

  /// \brief The camera pose at the time when this frame was observed.
  Eigen::Matrix4f camera_pose;

  /// \brief The relative pose to the previous frame in the track, if it could be computed.
  dynslam::utils::Option<Eigen::Matrix4d> relative_pose;

  TrackFrame(int frame_idx, const InstanceView& instance_view, const Eigen::Matrix4f camera_pose)
      : frame_idx(frame_idx), instance_view(instance_view), camera_pose(camera_pose) {}

  SUPPORT_EIGEN_FIELDS;
};

/// \brief A detected object's track through multiple frames.
/// Modeled as a series of detections, contained in the 'frames' field. Note that there can be
/// gaps in this list, due to frames where this particular object was not detected.
/// \todo In the long run, this class should be able to leverage a 3D reconstruction and something
///       like a Kalman Filter for motion tracking to predict an object's (e.g., car) pose in a
///       subsequent frame, in order to aid with tracking.
class Track {
 public:
  Track(int id) : id_(id), reconstruction(nullptr) {}

  virtual ~Track() {
    if (reconstruction.get() != nullptr) {
      fprintf(stderr, "Deleting track [%d] and its associated reconstruction!\n", id_);
    }
  }

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

  const std::vector<TrackFrame>& GetFrames() const { return frames_; }

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

  bool HasReconstruction() const { return reconstruction.get() != nullptr; }

  std::shared_ptr<dynslam::drivers::InfiniTamDriver>& GetReconstruction() {
    return reconstruction;
  }

  const std::shared_ptr<dynslam::drivers::InfiniTamDriver>& GetReconstruction() const {
    return reconstruction;
  }

  /// \brief Uses a series of goodness heuristics to establish whether the information contained in
  /// this track's frames is good enough for a 3D reconstruction.
  bool EligibleForReconstruction() const {
    // TODO(andrei): Moonshot---use a classifier to do this based on, e.g., track length, some
    // pose info, frame sizes, etc. Main challenge: how to get training data?

    // For now, use this simple heuristic: at least k frames in track.
    return GetSize() >= 6;
  }

  /// \brief Returns the relative pose of the specified frame w.r.t. the first one.
  dynslam::utils::Option<Eigen::Matrix4d> GetFramePose(size_t frame_idx) const;

  bool NeedsCleanup() const {
    return needs_cleanup_;
  }

  void SetNeedsCleanup(bool needs_cleanup) {
    this->needs_cleanup_ = needs_cleanup;
  }

 private:
  /// \brief A unique identifier for this particular track.
  int id_;
  std::vector<TrackFrame> frames_;

  /// \brief A pointer to a 3D reconstruction of the object in this track.
  /// Is set to `nullptr` if no reconstruction is available.
  std::shared_ptr<dynslam::drivers::InfiniTamDriver> reconstruction;

  /// \brief Whether the reconstruction is pending a full voxel decay iteration.
  bool needs_cleanup_;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_TRACK_H
