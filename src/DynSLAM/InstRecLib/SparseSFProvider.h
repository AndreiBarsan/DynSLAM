

#ifndef INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
#define INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H

#include <opencv/cv.h>
#include "../../libviso2/src/matcher.h"

#include <Eigen/Dense>

namespace instreclib {

using ViewPair = std::pair<cv::Mat1b*, cv::Mat1b*>;

/// \brief Contains the result of a (sparse) scene flow estimation (tuples of matches in the
/// left/right current/past frames (not 3D vectors yet).
/// Currently only supports libviso2-style data.
class SparseSceneFlow {
 public:
  std::vector<Matcher::p_match> matches;

//  Eigen::Matrix<float, Eigen::Dynamic, 8> matches;
};

/// \brief Interface for components which can compute sparse scene flow from a scene view.
class SparseSFProvider {
 public:
  virtual void ComputeSparseSF(const ViewPair &last_view, const ViewPair &current_view) = 0;

  virtual bool FlowAvailable() const = 0;

  /// \brief Returns the latest scene flow information computed using 'ComputeSparseSF'.
  /// \note This will not be available until at least two frames have been processed.
  virtual SparseSceneFlow& GetFlow() = 0;
};

}  // namespace instreclib

#endif  // INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
