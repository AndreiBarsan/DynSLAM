
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
                                           const Eigen::Vector3d &velo_2d_homo,
                                           float rendered_disp,
                                           float rendered_depth_m,
                                           float input_disp,
                                           float input_depth_m,
                                           float lidar_disp,
                                           int frame_width,
                                           int frame_height
) {
  measurement_count_++;
  const float ren_disp_delta = fabs(rendered_disp - lidar_disp);
  const float input_disp_delta = fabs(input_disp - lidar_disp);

  /// We want to compare the fusion and the input map only where they're both present, since
  /// otherwise the fusion covers a much larger area, so its evaluation is tougher.
  if (compare_on_intersection && (fabs(input_depth_m) < 1e-5 || fabs(rendered_depth_m) < 1e-5)) {
    input_stats_.missing++;
    rendered_stats_.missing++;
  } else {
    if (fabs(input_depth_m) < 1e-5) {
      input_stats_.missing++;
    } else {
      bool is_error = (kitti_style) ?
                      (input_disp_delta > delta_max && (input_disp_delta > 0.05 * lidar_disp)) :
                      (input_disp_delta > delta_max);
      if (is_error) {
        input_stats_.error++;
      } else {
        input_stats_.correct++;
      }
    }

    if (rendered_depth_m < 1e-5) {
      rendered_stats_.missing++;
    } else {
      bool is_error = (kitti_style) ?
                      (ren_disp_delta > delta_max && (ren_disp_delta > 0.05 * lidar_disp)) :
                      (ren_disp_delta > delta_max);
      if (is_error) {
        rendered_stats_.error++;
      } else {
        rendered_stats_.correct++;
      }
    }
  }
}

}
}
