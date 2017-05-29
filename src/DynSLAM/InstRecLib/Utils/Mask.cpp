

#include "Mask.h"

namespace instreclib {
namespace utils {

void Mask::Set(const Mask &rhs) {
  // Handle the bounding box
  this->bounding_box_ = rhs.bounding_box_;

  // Handle the detailed pixel-wise mask
  delete mask_data_;
  mask_data_ = new cv::Mat(*rhs.mask_data_);
}

}   // namespace utils
}   // namespace instreclib
