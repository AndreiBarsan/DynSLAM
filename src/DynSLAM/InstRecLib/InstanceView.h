

#ifndef INSTRECLIB_INSTANCEVIEW_H
#define INSTRECLIB_INSTANCEVIEW_H

#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "InstanceSegmentationResult.h"

namespace instreclib {
namespace reconstruction {

/// \brief Like ITMView, but associated with a particular object instance.
class InstanceView {
 public:
  InstanceView(const segmentation::InstanceDetection& instance_detection_,
               const std::shared_ptr<ITMLib::Objects::ITMView>& view_)
      : instance_detection_(instance_detection_), view_(view_) {}

  virtual ~InstanceView() {}

  ITMLib::Objects::ITMView* GetView() { return view_.get(); }

  const ITMLib::Objects::ITMView* GetView() const { return view_.get(); }

  instreclib::segmentation::InstanceDetection& GetInstanceDetection() {
    return instance_detection_;
  }

  const instreclib::segmentation::InstanceDetection& GetInstanceDetection() const {
    return instance_detection_;
  }

 private:
  /// \brief Holds label, mask, and bounding box information.
  instreclib::segmentation::InstanceDetection instance_detection_;

  /// \brief Holds the depth and RGB information about the object.
  std::shared_ptr<ITMLib::Objects::ITMView> view_;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCEVIEW_H
