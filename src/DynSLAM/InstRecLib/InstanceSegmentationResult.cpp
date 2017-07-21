
#include "InstanceSegmentationResult.h"

#include <iomanip>

namespace instreclib {
namespace segmentation {

using namespace std;

string InstanceDetection::GetClassName() const { return segmentation_dataset->labels[class_id]; }

ostream &operator<<(ostream &out, const InstanceDetection &detection) {
  out << detection.GetClassName() << " at " << detection.GetCopyBoundingBox() << ". "
      << "Probability: " << setprecision(4) << setw(6) << detection.class_probability << ".";
  return out;
}

ostream &operator<<(ostream &out, const InstanceSegmentationResult &segmentation_result) {
  if (segmentation_result.instance_detections.size() > 0) {
    out << "Detected " << segmentation_result.instance_detections.size() << " objects in the frame."
        << std::endl;
    for (const auto &instance : segmentation_result.instance_detections) {
      out << "\t " << instance << std::endl;
    }
  } else {
    out << "Nothing detected in the frame." << std::endl;
  }
  return out;
}

}  // namespace segmentation
}  // namespace instreclib
