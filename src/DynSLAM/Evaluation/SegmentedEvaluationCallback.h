#ifndef DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H
#define DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H

#include "EvaluationCallback.h"
#include "SegmentedCallback.h"

namespace dynslam {
namespace eval {

/// \brief Similar to 'EvaluationCallback', but computes separate error scores for static components
/// of the environment and (potentially dynamic) cars. Other potentially dynamic, but not
/// reconstructable, objects, such as cyclists and pedestrians are simply not evaluated.
class SegmentedEvaluationCallback : public SegmentedCallback {
 public:
  SegmentedEvaluationCallback(float delta_max, bool compare_on_intersection, bool kitti_style,
                              instreclib::segmentation::InstanceSegmentationResult *frame_segmentation,
                              instreclib::reconstruction::InstanceReconstructor *reconstructor
  )
      : SegmentedCallback(frame_segmentation, reconstructor),
        static_eval_(delta_max, compare_on_intersection, kitti_style),
        dynamic_eval_(delta_max, compare_on_intersection, kitti_style)
  {}

  void ProcessLidarPoint(int idx,
                         const Eigen::Vector3d &velo_2d_homo_px,
                         float rendered_disp,
                         float rendered_depth_m,
                         float input_disp,
                         float input_depth_m,
                         float lidar_disp,
                         int frame_width,
                         int frame_height) override;

  DepthEvaluation GetStaticEvaluation() {
    return static_eval_.GetEvaluation();
  }

  DepthEvaluation GetDynamicEvaluation() {
    return dynamic_eval_.GetEvaluation();
  }

  long GetSkippedLidarPoints() const {
    return skipped_lidar_points_;
  }

 private:
  EvaluationCallback static_eval_;
  EvaluationCallback dynamic_eval_;

};

}
}

#endif //DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H

