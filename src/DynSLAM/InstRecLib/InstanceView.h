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
  using FlowVector = std::vector<RawFlow, Eigen::aligned_allocator<RawFlow>>;

  InstanceView(const segmentation::InstanceDetection &instance_detection,
               const std::shared_ptr<ITMLib::Objects::ITMView> &view,
               const FlowVector &sparse_sf)
      : instance_detection_(instance_detection),
        view_(view),
        sparse_sf_(sparse_sf) {}

  virtual ~InstanceView() { }

  ITMLib::Objects::ITMView* GetView() { return view_.get(); }
  const ITMLib::Objects::ITMView* GetView() const { return view_.get(); }

  instreclib::segmentation::InstanceDetection& GetInstanceDetection() {
    return instance_detection_;
  }

  const instreclib::segmentation::InstanceDetection& GetInstanceDetection() const {
    return instance_detection_;
  }

  const FlowVector GetFlow() const {
    return sparse_sf_;
  }

  /// \brief Deallocates the view object and any GPU memory it may have allocated.
  void DiscardView() {
    view_.reset();
  }

 private:
  /// \brief Holds label, mask, and bounding box information.
  instreclib::segmentation::InstanceDetection instance_detection_;

  /// \brief Holds the depth and RGB information about the object.
  std::shared_ptr<ITMLib::Objects::ITMView> view_;

  /// \brief The scene scene flow data associated with this instance at this time.
  FlowVector sparse_sf_;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCEVIEW_H
