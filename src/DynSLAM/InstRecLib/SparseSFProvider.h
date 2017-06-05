

#ifndef INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
#define INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H

#include <opencv/cv.h>

namespace instreclib {

using ViewPair = std::pair<cv::Mat1b*, cv::Mat1b*>;

/// \brief Interface for components which can compute sparse scene flow from a scene view.
class SparseSFProvider {
 public:
  // TODO(andrei): Data holder for SF computation result.
  virtual void ComputeSparseSceneFlow(const ViewPair &last_view, const ViewPair &current_view) = 0;

 private:
};

}  // namespace instreclib

#endif  // INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
