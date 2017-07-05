

#ifndef INSTRECLIB_SEGMENTATIONPROVIDER_H
#define INSTRECLIB_SEGMENTATIONPROVIDER_H

#include <memory>

#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "InstanceSegmentationResult.h"

namespace instreclib {
namespace segmentation {

/// \brief Performs semantic segmentation on input frames.
class SegmentationProvider {
 public:
  virtual ~SegmentationProvider(){};

  /// \brief Performs semantic segmentation of the given frame.
  /// Usually uses only RGB data, but some segmentation pipelines may leverage depth as well.
  virtual std::shared_ptr<InstanceSegmentationResult> SegmentFrame(const cv::Mat3b &rgb) = 0;

  virtual cv::Mat3b *GetSegResult() = 0;

  virtual const cv::Mat3b *GetSegResult() const = 0;
};

}  // namespace segmentation
}  // namespace instreclib

#endif  // INSTRECLIB_SEGMENTATIONPROVIDER_H
