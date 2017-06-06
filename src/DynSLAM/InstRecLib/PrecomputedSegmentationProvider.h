

#ifndef INSTRECLIB_PRECOMPUTEDSEGMENTATIONPROVIDER_H
#define INSTRECLIB_PRECOMPUTEDSEGMENTATIONPROVIDER_H

#include <memory>

#include "InstanceSegmentationResult.h"
#include "SegmentationProvider.h"

namespace instreclib {
namespace segmentation {

/// \brief Reads pre-existing frame segmentations from the disk, instead of computing them
/// on-the-fly.
class PrecomputedSegmentationProvider : public SegmentationProvider {
 public:
  PrecomputedSegmentationProvider(const std::string &seg_folder)
      : seg_folder_(seg_folder), frame_idx_(0), dataset_used(&kPascalVoc2012), last_seg_preview_(nullptr) {
  }

  ~PrecomputedSegmentationProvider() override { delete last_seg_preview_; }

  std::shared_ptr<InstanceSegmentationResult> SegmentFrame(const cv::Mat4b &view) override;

  const cv::Mat4b *GetSegResult() const override;

  cv::Mat4b *GetSegResult() override;

 protected:
  /// \brief For the segmentation from the given file path, loads all available file containing
  /// detection information (class, bounding box, etc.) and instance masks.
  std::vector<InstanceDetection> ReadInstanceInfo(const std::string &base_img_fpath);

 private:
  std::string seg_folder_;
  int frame_idx_;
  const SegmentationDataset *dataset_used;
  cv::Mat4b *last_seg_preview_;

};

}  // namespace segmentation
}  // namespace instreclib

#endif  // INSTRECLIB_PRECOMPUTEDSEGMENTATIONPROVIDER_H
