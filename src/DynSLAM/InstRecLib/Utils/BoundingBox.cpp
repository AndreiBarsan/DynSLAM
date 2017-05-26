
#include "BoundingBox.h"

#include <cmath>

namespace InstRecLib {
	namespace Utils {
		using namespace std;

		ostream& operator<<(ostream& out, const BoundingBox& bounding_box) {
			out << "[(" << bounding_box.r.x0 << ", " << bounding_box.r.y0 << "), ("
				          << bounding_box.r.x1 << ", " << bounding_box.r.y1 << ")]";
			return out;
		}

		BoundingBox BoundingBox::IntersectWith(const BoundingBox &other) const {
			if (! this->Intersects(other)) {
				// Return an empty bounding box.
				// TODO(andrei): Make this more rigorous.
				return BoundingBox(0, 0, -1, -1);
			}

			int max_x0 = max(this->r.x0, other.r.x0);
			int min_x1 = min(this->r.x1, other.r.x1);
			int max_y0 = max(this->r.y0, other.r.y0);
			int min_y1 = min(this->r.y1, other.r.y1);

			return BoundingBox(max_x0, max_y0, min_x1, min_y1);
		}

		bool BoundingBox::Intersects(const BoundingBox &other) const {
			if (other.r.x0 < this->r.x1 && this->r.x0 < other.r.x1 && other.r.y0 < this->r.y1) {
				if (this->r.y0 < other.r.y1) {
					return true;
				}

				return false;
			}

			return false;
		}
	}
}

