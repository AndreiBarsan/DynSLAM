

#include "Mask.h"

namespace instreclib {
namespace utils {

void Mask::Clear() {
  if (mask_) {
    // TODO(andrei): Handle the mask in a more modern way!
    for (int row = 0; row < GetHeight(); ++row) {
      if (mask_[row]) {
        delete[] mask_[row];
      }
    }
    delete[] mask_;
  }
}

void Mask::Set(const Mask &rhs) {
  // Handle the bounding box
  this->bounding_box_ = rhs.bounding_box_;

  // Handle the detailed pixel-wise mask
  int height = rhs.GetHeight();
  int width = rhs.GetWidth();
  this->Clear();

  this->mask_ = new uint8_t *[height];
  for (int i = 0; i < height; ++i) {
    this->mask_[i] = new uint8_t[width];
    std::memcpy(this->mask_[i], rhs.mask_[i], width * sizeof(uint8_t));
  }
}

}   // namespace utils
}   // namespace instreclib
