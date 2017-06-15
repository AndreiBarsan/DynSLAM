

#ifndef INSTRECLIB_INSTANCEVIEW_H
#define INSTRECLIB_INSTANCEVIEW_H

#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "InstanceSegmentationResult.h"
#include "SparseSFProvider.h"
#include "../Utils.h"

namespace instreclib {
namespace reconstruction {

/// \brief Like ITMView, but associated with a particular object instance.
class InstanceView {
 public:
  InstanceView(const segmentation::InstanceDetection &instance_detection,
               const std::shared_ptr<ITMLib::Objects::ITMView> &view,
               const std::vector<RawFlow> &sparse_sf,
               const std::shared_ptr<dynslam::utils::Option<Eigen::Matrix4d>> motion_delta)
      : instance_detection_(instance_detection),
        view_(view),
        sparse_sf_(sparse_sf),
        motion_delta_(motion_delta) {}

  virtual ~InstanceView() { }

  ITMLib::Objects::ITMView* GetView() { return view_.get(); }

  const ITMLib::Objects::ITMView* GetView() const { return view_.get(); }

  instreclib::segmentation::InstanceDetection& GetInstanceDetection() {
    return instance_detection_;
  }

  const instreclib::segmentation::InstanceDetection& GetInstanceDetection() const {
    return instance_detection_;
  }

  const std::vector<RawFlow> GetFlow() const {
    return sparse_sf_;
  }

  /// \brief Whether the transform between the previous state and the current is known.
  bool HasRelativePose() const {
    return motion_delta_->IsPresent();
  }

  const Eigen::Matrix4d& GetRelativePose() const {
    return **motion_delta_;   // Dereference, then grab the object out of the Option<>.
  }

 private:
  /// \brief Holds label, mask, and bounding box information.
  instreclib::segmentation::InstanceDetection instance_detection_;

  /// \brief Holds the depth and RGB information about the object.
  std::shared_ptr<ITMLib::Objects::ITMView> view_;

  // TODO(andrei): We could probably get rid of this.
  /// \brief The scene scene flow data associated with this instance at this time.
  std::vector<RawFlow> sparse_sf_;

  /// \brief Motion delta transform from the previous frame to the current. Unavailable in the first
  ///        frame of a track, or in frames where motion could not be estimated.
  std::shared_ptr<const dynslam::utils::Option<Eigen::Matrix4d>> motion_delta_;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCEVIEW_H
