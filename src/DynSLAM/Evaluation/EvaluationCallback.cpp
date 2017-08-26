
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

DepthEvaluation EvaluationCallback::CreateDepthEvaluation(float delta_max,
                                                          long measurement_count,
                                                          bool kitti_style,
                                                          const Stats &rendered_stats,
                                                          const Stats &input_stats) {
  DepthResult rendered_result(measurement_count, rendered_stats.error, rendered_stats.missing,
                              rendered_stats.correct, rendered_stats.missing_separate);
  DepthResult input_result(measurement_count, input_stats.error, input_stats.missing,
                           input_stats.correct, input_stats.missing_separate);
  return DepthEvaluation(delta_max,
                         std::move(rendered_result),
                         std::move(input_result),
                         kitti_style);
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

  bool missing_input = (fabs(input_depth_m) < 1e-5);
  bool missing_rendered = (fabs(rendered_depth_m) < 1e-5);

  if (missing_input) {
    input_stats.missing_separate++;
  }
  if (missing_rendered) {
    rendered_stats.missing_separate++;
  }

  /// We want to compare the fusion and the input map only where they're both present, since
  /// otherwise the fusion covers a much larger area, so its evaluation is tougher.
  if (compare_on_intersection && (missing_input || missing_rendered)) {
    // TODO-LOW(andrei): Maybe get rid of the 'compare_on_intersection' flag and just count
    // the # of pixels absent in either the input or rendered depth in a separate metric.
    input_stats.missing++;
    rendered_stats.missing++;
  } else {
    if (missing_input) {
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

    if (missing_rendered) {
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
