
#include "InstanceSegmentationResult.h"

#include <iomanip>

namespace InstRecLib {
	namespace Segmentation {
		using namespace std;

		string InstanceDetection::GetClassName() const {
			return segmentation_dataset->labels[class_id];
		}

		ostream& operator<<(ostream& out, const InstanceDetection& detection) {
			out << detection.GetClassName() << " at " << detection.GetBoundingBox() << ". "
			    << "Probability: " << setprecision(4) << setw(6) << detection.class_probability << ".";
			return out;
		}
	}
}

