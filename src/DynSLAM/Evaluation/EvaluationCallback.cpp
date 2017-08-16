
#include "EvaluationCallback.h"

namespace dynslam {
namespace eval {

EvaluationCallback::EvaluationCallback(const float delta_max,
                                       const bool compare_on_intersection,
                                       const bool kitti_style)
    : delta_max(delta_max),
      compare_on_intersection(compare_on_intersection),
      kitti_style(kitti_style),
      input_stats_({}),
      rendered_stats_({}) {}

void EvaluationCallback::ProcessLidarPoint(int idx,
                                           const Eigen::Vector3d &velo_2d_homo_px,
                                           float rendered_disp,
                                           float rendered_depth_m,
                                           float input_disp,
                                           float input_depth_m,
                                           float lidar_disp,
                                           int frame_width,
                                           int frame_height
) {
  measurement_count_++;
  ComputeAccuracy(rendered_disp, rendered_depth_m, input_disp, input_depth_m, lidar_disp,
                  input_stats_, rendered_stats_);
}

void EvaluationCallback::ComputeAccuracy(float rendered_disp,
                                         float rendered_depth_m,
                                         float input_disp,
                                         float input_depth_m,
                                         float lidar_disp,
                                         Stats &input_stats,
                                         Stats &rendered_stats
) {
  const float ren_disp_delta = fabs(rendered_disp - lidar_disp);
  const float input_disp_delta = fabs(input_disp - lidar_disp);

  /// We want to compare the fusion and the input map only where they're both present, since
  /// otherwise the fusion covers a much larger area, so its evaluation is tougher.
  if (compare_on_intersection && (fabs(input_depth_m) < 1e-5 || fabs(rendered_depth_m) < 1e-5)) {
    input_stats.missing++;
    rendered_stats.missing++;
  } else {
    if (fabs(input_depth_m) < 1e-5) {
      input_stats.missing++;
    } else {
      bool is_error = (kitti_style) ?
                      (input_disp_delta > delta_max && (input_disp_delta > 0.05 * lidar_disp)) :
                      (input_disp_delta > delta_max);
      if (is_error) {
        input_stats.error++;
      } else {
        input_stats.correct++;
      }
    }

    if (rendered_depth_m < 1e-5) {
      rendered_stats.missing++;
    } else {
      bool is_error = (kitti_style) ?
                      (ren_disp_delta > delta_max && (ren_disp_delta > 0.05 * lidar_disp)) :
                      (ren_disp_delta > delta_max);
      if (is_error) {
        rendered_stats.error++;
      } else {
        rendered_stats.correct++;
      }
    }
  }
}

}
}
