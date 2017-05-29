

#ifndef INSTRECLIB_MASK_H
#define INSTRECLIB_MASK_H

#include <cstdint>
#include <cstring>

#include <opencv/cv.h>

#include "BoundingBox.h"

namespace instreclib {
namespace utils {

// TODO(andrei): Make this class CUDA-proof.
// TODO(andrei): CUDA+CPU methods for computing mask intersection and the
// overlap area.
//
/// \brief An image mask consisting of a bounding box and a detailed boolean
/// mask array.
/// Used to model semantic segmentation results.
/// A bounding box's (0, 0) is the top-left, in accordance with the rest of
/// InfiniTAM.
class Mask {
 public:
  /// \brief Initializes this object, taking ownership of the given mask*.
  Mask(const BoundingBox& bounding_box, cv::Mat* mask)
      : bounding_box_(bounding_box), mask_data_(mask) {}

  /// \brief Initializes this object, copying the other's data.
  Mask(const Mask& other) { this->Set(other); }

  /// \brief Assigns to this object, copying the other's data.
  Mask& operator=(const Mask& other) {
    this->Set(other);
    return *this;
  }

  virtual ~Mask() { delete mask_data_; }

  int GetWidth() const { return bounding_box_.GetWidth(); }

  int GetHeight() const { return bounding_box_.GetHeight(); }

  const BoundingBox& GetBoundingBox() const { return bounding_box_; }

  BoundingBox& GetBoundingBox() { return bounding_box_; }

  const cv::Mat* GetMask() const { return mask_data_; }

 private:
  BoundingBox bounding_box_;

  /// 2D binary matrix indicating the instance's occupancy pixels, within the bounding box.
  /// The mask has the same dimensions as the bounding box.
  cv::Mat *mask_data_;

  void Set(const Mask& rhs);
};

}   // namespace utils
}   // namespace instreclib

#endif  // INSTRECLIB_MASK_H
