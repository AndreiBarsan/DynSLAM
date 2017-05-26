

#ifndef INFINITAM_BOUNDINGBOX_H
#define INFINITAM_BOUNDINGBOX_H

#include <iostream>

namespace InstRecLib {
	namespace Utils {

		/// \brief A rectangular bounding box, with <b>inclusive coordinates</b>.
		struct BoundingBox {
			// Allow addressing the points both together, as an int array, as well as individually.
			union {
				struct {
					int x0;
					int y0;
					int x1;
					int y1;
				} r;
				int points[4];
			};

			int GetWidth() const {
				return r.x1 - r.x0 + 1;
			}

			int GetHeight() const {
				return r.y1 - r.y0 + 1;
			}

			BoundingBox() : BoundingBox(0, 0, 0, 0) { }

			BoundingBox(const int* const points)
					: BoundingBox(points[0], points[1], points[2], points[3]) { }

			BoundingBox(int x0, int y0, int x1, int y1) {
				this->r.x0 = x0;
				this->r.x1 = x1;
				this->r.y0 = y0;
				this->r.y1 = y1;
			}

			bool ContainsPoint(const int x, const int y) const {
				return (x >= this->r.x0 && x <= this->r.x1 && y >= this->r.y0 && y <= this->r.y1);
			}

			bool Intersects(const BoundingBox& other) const;

			/// \brief Computes the intersection between this and the other bounding box.
			/// \return The resulting box, or a (0, 0, 0, 0)-sized (empty) bounding box otherwise.
			BoundingBox IntersectWith(const BoundingBox& other) const;

			int GetArea() const {
				return GetWidth() * GetHeight();
			}
		};

		std::ostream& operator<<(std::ostream& out, const BoundingBox& bounding_box);
	}
}


#endif //INFINITAM_BOUNDINGBOX_H
