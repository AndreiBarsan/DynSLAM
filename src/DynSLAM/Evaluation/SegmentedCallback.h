#ifndef DYNSLAM_SEGMENTEDCALLBACK_H
#define DYNSLAM_SEGMENTEDCALLBACK_H

#include "ILidarEvalCallback.h"
#include "EvaluationCallback.h"

namespace dynslam {
namespace eval {

/// \brief Performs an action depending on whether a LIDAR point corresponds to a static map area,
///        to a dynamic object being reconstructed, or neither.
class SegmentedCallback : public ILidarEvalCallback {
 public:
  enum LidarAssociation {
    kStaticMap,
    kDynamicReconstructed,
    kNeither
  };

 public:
  SegmentedCallback(InstanceSegmentationResult *frame_segmentation_,
                    InstanceReconstructor *reconstructor_)
      : frame_segmentation_(frame_segmentation_), reconstructor_(reconstructor_) {}

 protected:
  instreclib::segmentation::InstanceSegmentationResult *frame_segmentation_;
  instreclib::reconstruction::InstanceReconstructor* reconstructor_;
  long skipped_lidar_points_ = 0;

  LidarAssociation GetPointAssociation(const Eigen::Vector3d &velo_2d_homo_px);
};

}
}

#endif //DYNSLAM_SEGMENTEDCALLBACK_H

