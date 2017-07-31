

#ifndef INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
#define INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H

#include <opencv/cv.h>
#include "../../libviso2/src/matcher.h"
#include "../../DynSLAM/Defines.h"

#include <Eigen/Dense>

namespace instreclib {

using ViewPair = std::pair<cv::Mat1b*, cv::Mat1b*>;

struct RawFlow {
  Eigen::Vector2f curr_left;
  // Feature index used by the underlying scene flow system for matching (e.g., in libviso2).
  int32_t curr_left_idx;
  Eigen::Vector2f curr_right;
  int32_t curr_right_idx;

  Eigen::Vector2f prev_left;
  int32_t prev_left_idx;
  Eigen::Vector2f prev_right;
  int32_t prev_right_idx;

  RawFlow(float c_left_x,  float c_left_y,  int c_left_idx,
          float c_right_x, float c_right_y, int c_right_idx,
          float p_left_x,  float p_left_y,  int p_left_idx,
          float p_right_x, float p_right_y, int p_right_idx)
      : curr_left(c_left_x, c_left_y),
        curr_left_idx(c_left_idx),
        curr_right(c_right_x, c_right_y),
        curr_right_idx(c_right_idx),
        prev_left(p_left_x, p_left_y),
        prev_left_idx(p_left_idx),
        prev_right(p_right_x, p_right_y),
        prev_right_idx(p_right_idx) { }

  SUPPORT_EIGEN_FIELDS;
};

/// \brief Contains the result of a (sparse) scene flow estimation (tuples of matches in the
///        left/right current/past frames (not 3D vectors yet).
class SparseSceneFlow {
 public:
  std::vector<RawFlow, Eigen::aligned_allocator<RawFlow>> matches;
};

/// \brief Interface for components which can compute sparse scene flow from a scene view.
class SparseSFProvider {
 public:
  virtual ~SparseSFProvider() = default;
  SparseSFProvider() = default;
  SparseSFProvider(const SparseSFProvider&) = default;
  SparseSFProvider(SparseSFProvider&&) = default;
  SparseSFProvider& operator=(const SparseSFProvider&) = default;
  SparseSFProvider& operator=(SparseSFProvider&&) = default;

  virtual void ComputeSparseSF(const ViewPair &last_view, const ViewPair &current_view) = 0;

  virtual bool FlowAvailable() const = 0;

  /// \brief Returns the latest scene flow information computed using 'ComputeSparseSF'.
  /// \note This will not be available until at least two frames have been processed.
  virtual SparseSceneFlow& GetFlow() = 0;

  // TODO-LOW(andrei): Get rid of this.
  // Another hack until we create a dedicated ITM tracker which calls into this...
  // Returns the latest visual odometry estimate.
  virtual Eigen::Matrix4f GetLatestMotion() const = 0;

  // Hacky proxy for using viso's sf utilities for motion estimation in the inst. rec.
  virtual std::vector<double> ExtractMotion(
      const std::vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow,
      const std::vector<double> &initial_estimate
  ) const = 0;
};

}  // namespace instreclib

#endif  // INSTRECLIB_SPARSESCENEFLOWCOMPONENT_H
