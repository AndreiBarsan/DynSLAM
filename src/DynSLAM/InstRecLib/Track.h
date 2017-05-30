

#ifndef INSTRECLIB_TRACK_H
#define INSTRECLIB_TRACK_H

#include <vector>
#include "InstanceSegmentationResult.h"
#include "InstanceView.h"
#include "../InfiniTamDriver.h"

namespace instreclib {
namespace reconstruction {

/// \brief One frame of an instance track (InstRecLib::Reconstruction::Track).
struct TrackFrame {
  int frame_idx;
  InstanceView instance_view;

  // Only active if inside a Track. TODO(andrei): Implement.
  TrackFrame* previous;
  TrackFrame* next;

 public:
  TrackFrame(int frame_idx, const InstanceView& instance_view)
      : frame_idx(frame_idx), instance_view(instance_view), previous(nullptr), next(nullptr) {}
};

/// \brief A detected object's track through multiple frames.
/// Modeled as a series of detections, contained in the 'frames' field. Note that there can be
/// gaps in this list, due to frames where this particular object was not detected.
class Track {
 private:
  /// \brief A unique identifier for this particular track.
  int id_;
  std::vector<TrackFrame> frames_;

  /// \brief A pointer to a 3D reconstruction of the object in this track.
  /// Is set to `nullptr` if no reconstruction is available.
  std::shared_ptr<dynslam::drivers::InfiniTamDriver> reconstruction;

 public:
  Track(int id) : id_(id), reconstruction(nullptr) {}
  virtual ~Track() { }

  /// \brief Evaluates how well this new frame would fit the existing track.
  /// \returns A goodness score between 0 and 1, where 0 means the new frame
  /// would not match
  /// this track at all, and 1 would be a perfect match.
  float ScoreMatch(const TrackFrame& new_frame) const;

  void AddFrame(const TrackFrame& new_frame) { frames_.push_back(new_frame); }

  size_t GetSize() const { return frames_.size(); }

  TrackFrame& GetLastFrame() { return frames_.back(); }

  const TrackFrame& GetLastFrame() const { return frames_.back(); }

  int GetStartTime() const { return frames_.front().frame_idx; }

  int GetEndTime() const { return frames_.back().frame_idx; }

  const std::vector<TrackFrame>& GetFrames() const { return frames_; }

  int GetId() const { return id_; }

  /// \brief Draws a visual representation of this feature track.
  /// \example For an object first seen in frame 11, then in frames 12, 13, and
  /// 16, this
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
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_TRACK_H
