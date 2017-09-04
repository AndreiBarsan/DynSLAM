

#ifndef INSTRECLIB_INSTANCETRACKER_H
#define INSTRECLIB_INSTANCETRACKER_H

#include <cassert>
#include <list>
#include <map>
#include <Eigen/StdVector>

#include "InstanceSegmentationResult.h"
#include "InstanceView.h"
#include "Track.h"

// TODO(andrei): Consider splitting this file up into multiple class files.
namespace instreclib {
namespace reconstruction {

/// \brief Minimum overlap score required to add a new frame to an existing feature track.
/// Between 0.0 and 1.0.
const float kTrackScoreThreshold = 0.10f;

/// \brief Default age of the last frame in an object track after which we discard it.
/// The smaller this is, the less memory the system uses, but the likelier it is to fragment object
/// reconstructions into multiple volumes.
const int kDefaultInactiveFrameThreshold = 50;

/// \brief Tracks instances over time by associating multiple isolated detections.
/// Currently implemented by best-overlap using the bounding boxes.
class InstanceTracker {
 private:
  using TrackMap = std::map<int, Track, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Track>>>;
  TrackMap id_to_active_track_;

  /// \brief The maximum age of the latest frame in a track, before it is discarded.
  /// The higher this is, the more tracks are held in memory.
  int inactive_frame_threshold_;

  /// \brief The total number of tracks seen, including both active and pruned tracks.
  int track_count_;

 protected:
  /// \brief Constant meant to be returned by track matching code when no match is found.
  static constexpr std::pair<Track*, float> kNoBestTrack = std::pair<Track*, float>(nullptr, 0.0f);

  /// \brief Finds the most likely track for the given frame, if it exists.
  /// \return The best matching track, and the [0-1] match quality score. If no
  /// tracks are
  /// available, kNoTrack is returned.
  std::pair<Track*, float> FindBestTrack(const TrackFrame& track_frame);

  /// \brief Assign the detections to the best matching tracks.
  /// \note Mutates the `new_detections` input list, removing the matched
  /// detections.
  void AssignToTracks(std::list<TrackFrame, Eigen::aligned_allocator<TrackFrame>>& new_detections);

  /// \brief Removes tracks which have not been active in the past k frames.
  void PruneTracks(int current_frame_idx);

 public:
  InstanceTracker()
      : id_to_active_track_(TrackMap()),
        inactive_frame_threshold_(kDefaultInactiveFrameThreshold),
        track_count_(0) {}

  /// \brief Associates the new detections with existing tracks, or creates new ones.
  /// \param new_detections The instances detected in the current frame.
  void ProcessInstanceViews(
      int frame_idx,
      const std::vector<InstanceView, Eigen::aligned_allocator<InstanceView>>& new_detections,
      const Eigen::Matrix4f current_camera_pose
  );

  /// \see track_count_
  int GetTotalTrackCount() const { return track_count_; }

  int GetActiveTrackCount() const { return static_cast<int>(id_to_active_track_.size()); }

  /// \brief Checks whether a track for the specified object ID is available as an active track.
  bool HasTrack(int id) const {
    return id_to_active_track_.find(id) != id_to_active_track_.cend();
  }

  const Track& GetTrack(int id) const {
    return id_to_active_track_.at(id);
  }

  Track& GetTrack(int id) {
    return id_to_active_track_.at(id);
  }

  const TrackMap& GetActiveTracks() const {
    return id_to_active_track_;
  }

  const Track& GetTrackAtPoint(int x, int y, int current_frame_idx) const {
    for(auto it = std::begin(id_to_active_track_); it != std::end(id_to_active_track_); ++it) {
      const Track& track = it->second;
      if (track.GetLastFrame().frame_idx == (current_frame_idx - 1) &&
          track.GetLastFrame().instance_view.GetInstanceDetection().copy_mask->ContainsPoint(x, y)
      ) {
        return track;
      }
    }

    throw std::runtime_error(
        dynslam::utils::Format("Unable to find a track containing the point (%d, %d).", x, y));
  }
};

}  // namespace segmentation
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCETRACKER_H
