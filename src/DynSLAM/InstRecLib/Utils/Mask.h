

#ifndef INFINITAM_MASK_H
#define INFINITAM_MASK_H

#include <cstdint>
#include <cstring>

#include "BoundingBox.h"


namespace InstRecLib {
	namespace Utils {

		// TODO(andrei): Make this class CUDA-proof.
		// TODO(andrei): CUDA+CPU methods for computing mask intersection and the overlap area.
		//
		/// \brief An image mask consisting of a bounding box and a detailed boolean mask array.
		/// Used to model semantic segmentation results.
		/// A bounding box's (0, 0) is the top-left, in accordance with the rest of InfiniTAM.
		class Mask {
		private:
			BoundingBox bounding_box_;

			/// 2D binary array indicating the instance pixels, within the bounding box.
			/// It has the same dimensions as the bounding box.
			uint8_t **mask_;

			void Clear() {
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

			void Set(const Mask& rhs) {
				// Handle the bounding box
				this->bounding_box_ = rhs.bounding_box_;

				// Handle the detailed pixel-wise mask
				int height = rhs.GetHeight();
				int width = rhs.GetWidth();
				this->Clear();

				this->mask_ = new uint8_t*[height];
				for(int i = 0; i < height; ++i) {
					this->mask_[i] = new uint8_t[width];
					std::memcpy(this->mask_[i], rhs.mask_[i], width * sizeof(uint8_t));
				}
			}

		public:
			/// \brief Initializes this object, taking ownership of the given mask**.
			Mask(const BoundingBox& bounding_box, uint8_t** mask)
					: bounding_box_(bounding_box), mask_(mask) { }

			/// \brief Initializes this object, copying the other's data.
			Mask(const Mask& other) {
				this->Set(other);
			}

			/// \brief Assigns to this object, copying the other's data.
			Mask& operator=(const Mask& other) {
				this->Set(other);
				return *this;
			}

			virtual ~Mask() {
				this->Clear();
			}

			int GetWidth() const {
				return bounding_box_.GetWidth();
			}

			int GetHeight() const {
				return bounding_box_.GetHeight();
			}

			const BoundingBox& GetBoundingBox() const {
				return bounding_box_;
			}

			BoundingBox& GetBoundingBox() {
				return bounding_box_;
			}

			uint8_t** GetMask() {
				return mask_;
			}
		};

	}
}



#endif //INFINITAM_MASK_H
