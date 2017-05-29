

#ifndef INSTRECLIB_MASK_H
#define INSTRECLIB_MASK_H

#include <cstdint>
#include <cstring>

#include <opencv/cv.h>
#include <opencv/highgui.h>

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

  const cv::Mat* GetMaskData() const { return mask_data_; }

  /// \brief Resizes the mask and its bounding box, maintaining its aspect ratio.
  /// \param amount A value greater than zero. Values smaller than one cause the mask to shrink in
  /// size, while those greater than one increase its size.
  void Rescale(float amount) {
    int old_width = bounding_box_.GetWidth();
    int old_height = bounding_box_.GetHeight();

    int new_width = static_cast<int>(old_width * amount);
    int new_height = static_cast<int>(old_height * amount);

    int delta_width = new_width - old_width;
    int delta_height = new_height - old_height;

    int new_x0 = bounding_box_.r.x0 - static_cast<int>(floor(delta_width / 2.0));
    int new_y0 = bounding_box_.r.y0 - static_cast<int>(floor(delta_height / 2.0));
    int new_x1 = bounding_box_.r.x1 + static_cast<int>(ceil(delta_width / 2.0));
    int new_y1 = bounding_box_.r.y1 + static_cast<int>(ceil(delta_height / 2.0));

    cv::Size new_size(new_width, new_height);
    cv::Mat *tmp = new cv::Mat(new_size, CV_8UC1);
    cv::resize(*mask_data_, *tmp, tmp->size());

    bounding_box_ = BoundingBox(new_x0, new_y0, new_x1, new_y1);
    assert(bounding_box_.GetWidth() == new_width);
    assert(bounding_box_.GetHeight() == new_height);

    delete mask_data_;
    mask_data_ = tmp;
  }

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
