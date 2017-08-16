#ifndef DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H
#define DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H

#include "EvaluationCallback.h"

namespace dynslam {
namespace eval {

/// \brief Similar to 'EvaluationCallback', but computes separate error scores for static components
/// of the environment and (potentially dynamic) cars. Other potentially dynamic, but not
/// reconstructable, objects, such as cyclists and pedestrians are simply not evaluated.
class SegmentedEvaluationCallback : public EvaluationCallback {
 public:
  SegmentedEvaluationCallback(float delta_max, bool compare_on_intersection, bool kitti_style,
                              instreclib::segmentation::InstanceSegmentationResult *frame_segmentation
  )
      : input_stats_static_({}),
        rendered_stats_static_({}),
        input_stats_dynamic_({}),
        rendered_stats_dynamic_({}),
        frame_segmentation(frame_segmentation),
        EvaluationCallback::EvaluationCallback(delta_max, compare_on_intersection, kitti_style) {}

  void ProcessLidarPoint(int idx,
                         const Eigen::Vector3d &velo_2d_homo_px,
                         float rendered_disp,
                         float rendered_depth_m,
                         float input_disp,
                         float input_depth_m,
                         float lidar_disp,
                         int frame_width,
                         int frame_height) override;

  DepthEvaluation GetEvaluation() override;

 private:
  Stats input_stats_static_;
  Stats rendered_stats_static_;
  Stats input_stats_dynamic_;
  Stats rendered_stats_dynamic_;

  instreclib::segmentation::InstanceSegmentationResult *frame_segmentation;

  long measurement_count_static_ = 0;
  long measurement_count_dynamic_ = 0;
};

}
}

#endif //DYNSLAM_SEGMENTEDEVALUATIONCALLBACK_H

