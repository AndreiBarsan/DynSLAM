#ifndef DYNSLAM_EVALUATIONCALLBACK_H
#define DYNSLAM_EVALUATIONCALLBACK_H

#include "ILidarEvalCallback.h"
#include "Evaluation.h"

namespace dynslam {
namespace eval {

/// \brief Computes per-frame accuracy scores for comparing the input accuracy to the reconstruction
///        accuracy and the LIDAR ground truth.
class EvaluationCallback : public ILidarEvalCallback {
 public:
  const float delta_max;
  const bool compare_on_intersection;
  const bool kitti_style;

  EvaluationCallback(float delta_max,
                     bool compare_on_intersection,
                     bool kitti_style);

  void ProcessLidarPoint(int idx,
                         const Eigen::Vector3d &velo_2d_homo,
                         float rendered_disp,
                         float rendered_depth_m,
                         float input_disp,
                         float input_depth_m,
                         float lidar_disp,
                         int frame_width,
                         int frame_height) override;

  /// \brief Builds an aggregate evaluation object from the stats gathered by the object.
  /// \note Should be used *after* the data gets populated by 'Evaluation::EvaluateDepth'.
  DepthEvaluation GetEvaluation() {
    DepthResult rendered_result(measurement_count_, rendered_stats_.error, rendered_stats_.missing, rendered_stats_.correct);
    DepthResult input_result(measurement_count_, input_stats_.error, input_stats_.missing, input_stats_.correct);

    return DepthEvaluation(delta_max,
                           std::move(rendered_result),
                           std::move(input_result),
                           kitti_style);
  }

 private:
  Stats input_stats_;
  Stats rendered_stats_;

  long measurement_count_ = 0;
};

}
}

#endif //DYNSLAM_EVALUATIONCALLBACK_H
