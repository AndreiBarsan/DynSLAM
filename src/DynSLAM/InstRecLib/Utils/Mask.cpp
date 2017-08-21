

#include "Mask.h"

namespace instreclib {
namespace utils {

using namespace std;

void Mask::Set(const Mask &rhs) {
  // Handle the bounding box
  this->bounding_box_ = rhs.bounding_box_;

  // Handle the detailed pixel-wise mask
  if (nullptr != mask_data_) {
    delete mask_data_;
  }
  mask_data_ = new cv::Mat1b(*rhs.mask_data_);
}

void Mask::Rescale(float amount) {
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
  cv::Mat1b *tmp = new cv::Mat1b(new_size);
  cv::resize(*mask_data_, *tmp, tmp->size());

  bounding_box_ = BoundingBox(new_x0, new_y0, new_x1, new_y1);
  assert(bounding_box_.GetWidth() == new_width);
  assert(bounding_box_.GetHeight() == new_height);

  delete mask_data_;
  mask_data_ = tmp;
}

}   // namespace utils
}   // namespace instreclib
