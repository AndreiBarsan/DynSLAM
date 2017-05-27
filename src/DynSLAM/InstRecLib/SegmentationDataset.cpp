
#include "SegmentationDataset.h"

namespace instreclib {
namespace segmentation {

using namespace std;

map<string, int> labels_to_id_map(const vector<string> labels) {
  map<string, int> result;
  for (size_t i = 0; i < labels.size(); ++i) {
    result[labels[i]] = static_cast<int>(i);
  }
  return result;
};

}  // namespace segmentation
}  // namespace instreclib
