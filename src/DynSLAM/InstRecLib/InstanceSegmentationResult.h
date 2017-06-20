

#ifndef INSTRECLIB_SEGMENTATIONRESULT_H
#define INSTRECLIB_SEGMENTATIONRESULT_H

#include "SegmentationDataset.h"
#include "Utils/Mask.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace instreclib {
namespace segmentation {

class InstanceSegmentationResult;

// TODO(andrei): Consider storing an entire class-conditional distribution in each detection object.

/// \brief Describes a single object instance detected semantically in an input frame.
/// This is a component of InstanceSegmentationResult.
class InstanceDetection {
 public:
  /// This detection is the highest-likelihood class from a proposal region.
  /// This is its probability. It's usually pretty high (>0.9), but can still be useful for various
  /// sanity checks.
  float class_probability;

  /// Class identifier. Depends on the class labels used by the segmentation pipeline, specified in
  /// the associated `SegmentationDataset`.
  int class_id;

  /// \brief 2D mask of this detection in its source image frame.
  std::shared_ptr<instreclib::utils::Mask> mask;

  /// \brief Smaller, more conservative mask used for scene-flow association.
  /// Allows faster and more robust computation of object motion by reducing the amount of outlier
  /// scene flow vectors associated with the object.
  std::shared_ptr<instreclib::utils::Mask> conservative_mask;

  /// \brief The dataset associated with this instance's detection. Contains information such as
  /// mappings from class IDs to class names.
  const SegmentationDataset* segmentation_dataset;

  std::string GetClassName() const;

  instreclib::utils::BoundingBox& GetBoundingBox() { return mask->GetBoundingBox(); }

  const instreclib::utils::BoundingBox& GetBoundingBox() const { return mask->GetBoundingBox(); }

  /// \brief Initializes the detection with bounding box, class, and mask information.
  /// \note This object takes ownership of `mask` and `conservative_mask`.
  InstanceDetection(float class_probability,
                    int class_id,
                    std::shared_ptr<instreclib::utils::Mask> mask,
                    std::shared_ptr<instreclib::utils::Mask> conservative_mask,
                    const SegmentationDataset* segmentation_dataset)
      : class_probability(class_probability),
        class_id(class_id),
        mask(mask),
        conservative_mask(conservative_mask),
        segmentation_dataset(segmentation_dataset) {}

  virtual ~InstanceDetection() {}
};

/// \brief Supports pretty-printing segmentation result objects.
std::ostream& operator<<(std::ostream& out, const InstanceDetection& detection);

/// \brief The result of performing instance-aware semantic segmentation on an input frame.
struct InstanceSegmentationResult {
  /// \brief Specifies the dataset metadata, such as the labels, which are used by the segmentation.
  const SegmentationDataset* segmentation_dataset;

  /// \brief All object instances detected in a frame.
  std::vector<InstanceDetection> instance_detections;

  /// \brief The total time taken to compute this result, expressed in nanoseconds.
  long inference_time_ns;

  InstanceSegmentationResult(const SegmentationDataset* segmentation_dataset,
                             const std::vector<InstanceDetection>& instance_detections,
                             long inference_time_ns)
      : segmentation_dataset(segmentation_dataset),
        instance_detections(instance_detections),
        inference_time_ns(inference_time_ns) {}
};

/// \brief Pretty prints the result of a semantic segmentation.
std::ostream& operator<<(std::ostream& out, const InstanceSegmentationResult& segmentation_result);

}  // namespace segmentation
}  // namespace instreclib

#endif  // INSTRECLIB_SEGMENTATIONRESULT_H
