
#ifndef DYNSLAM_EVALUATION_H
#define DYNSLAM_EVALUATION_H

#include "../DynSlam.h"
#include "../Input.h"
#include "Velodyne.h"

namespace dynslam {
  class DynSlam;
}

namespace dynslam {
namespace eval {

/// \brief Stores the result of comparing a computed depth with a LIDAR ground truth.
struct DepthEvaluation {
  // Parameters
  const int frame_idx;
  const std::string dataset_id;
  const int delta_max;
  const float max_depth_meters;

  //Results
  const long measurement_count;
  const long error_count;

  const double error_mean;
  const double error_variance;

  DepthEvaluation(const int frame_idx,
                  const string &dataset_id,
                  const int delta_max,
                  const float max_depth_meters,
                  const long measurement_count,
                  const long error_count,
                  const double error_mean,
                  const double error_variance)
      : frame_idx(frame_idx),
        dataset_id(dataset_id),
        delta_max(delta_max),
        max_depth_meters(max_depth_meters),
        measurement_count(measurement_count),
        error_count(error_count),
        error_mean(error_mean),
        error_variance(error_variance) {}

  double GetCorrectPixelRatio() const {
    long correct_pixel_count = measurement_count - error_count;
    return static_cast<double>(correct_pixel_count) / measurement_count;
  }
};

/// \brief Main class handling the quantitative evaluation of the DynSLAM system.
class Evaluation {

 public:
  Evaluation(const std::string &dataset_root, const Input::Config &input_config,
             const Eigen::Matrix4f &velodyne_to_rgb, const Eigen::MatrixXf &left_cam_projection)
    : velodyne(new Velodyne(
        utils::Format("%s/%s", dataset_root.c_str(), input_config.velodyne_folder.c_str()),
        input_config.velodyne_fname_format,
        velodyne_to_rgb,
        left_cam_projection))
  { }

  /// \brief Supermethod in charge of all per-frame evaluation metrics.
  void EvaluateFrame(Input *input, DynSlam *dyn_slam);

  virtual ~Evaluation() {
    delete velodyne;
  }

  Velodyne* GetVelodyne() {
    return velodyne;
  }

  const Velodyne* GetVelodyne() const {
    return velodyne;
  }

 private:
  Velodyne *velodyne;

};

}
}

#endif //DYNSLAM_EVALUATION_H
