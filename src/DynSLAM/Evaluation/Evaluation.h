
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

class ICsvSerializable {
 public:
  /// \brief Should return the field names in the same order as GetData, without a newline.
  virtual std::string GetHeader() const = 0;
  virtual std::string GetData() const = 0;
};

struct DepthResult : public ICsvSerializable {
  const long measurement_count;
  const long error_count;
  const long missing_count;
  const long correct_count;

  DepthResult(const long measurement_count,
              const long error_count,
              const long missing_count,
              const long correct_count)
      : measurement_count(measurement_count),
        error_count(error_count),
        missing_count(missing_count),
        correct_count(correct_count) {
    assert(measurement_count == (error_count + missing_count + correct_count));
  }

  /// \brief Returns the ratio of correct pixels in this depth evaluation result.
  /// \param include_missing Whether to count pixels with no depth data in the evaluated depth map
  ///                        as incorrect. Formula: ratio = (correct) / (measurements)
  double GetCorrectPixelRatio(bool include_missing) const {
    if (include_missing) {
      return static_cast<double>(correct_count) / measurement_count;
    } else {
      return static_cast<double>(measurement_count - error_count)
          / (measurement_count - missing_count);
    }
  }

  string GetHeader() const override {
    return "measurements_count, error_count, missing_count, correct_count";
  }

  string GetData() const override {
    return utils::Format("%d, %d, %d, %d",
                         measurement_count,
                         error_count,
                         missing_count,
                         correct_count);
  }
};

struct DepthEvaluationMeta {
  const int frame_idx;
  const std::string dataset_id;

  DepthEvaluationMeta(const int frame_idx, const string &dataset_id)
      : frame_idx(frame_idx), dataset_id(dataset_id) {}
};

/// \brief Stores the result of comparing a computed depth with a LIDAR ground truth.
struct DepthEvaluation {
  // Parameters
  const DepthEvaluationMeta meta;
  const int delta_max;
  const float max_depth_meters;

  /// \brief Results for the depth map synthesized from engine.
  const DepthResult fused_result;

  /// \brief Results for the depth map received as input.
  const DepthResult input_result;

  DepthEvaluation(const DepthEvaluationMeta &meta,
                  const int delta_max,
                  const float max_depth_meters,
                  const DepthResult &fused_result,
                  const DepthResult &input_result)
      : meta(meta),
        delta_max(delta_max),
        max_depth_meters(max_depth_meters),
        fused_result(fused_result),
        input_result(input_result) {}
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
      left_cam_projection)) {}

  /// \brief Supermethod in charge of all per-frame evaluation metrics.
  void EvaluateFrame(Input *input, DynSlam *dyn_slam);

  /// TODO method to eval map depth vs. input depth vs. lidar on arbitrary frame.

  std::vector<DepthEvaluation> EvaluateFrame(int frame_idx, Input *input, DynSlam *dyn_slam);

  static uint depth_delta(uchar computed_depth, uchar ground_truth_depth) {
    return static_cast<uint>(abs(
        static_cast<int>(computed_depth) - static_cast<int>(ground_truth_depth)));
  }

  /// \brief Compares a fused depth map and input depth map to the corresponding LIDAR pointcloud,
  ///        which is considered to be the ground truth.
  ///
  /// Projects each LIDAR point (expressed in the LIDAR's reference frame) into the camera's frame,
  /// and compares its depth to that of both the fused depth map (rendered_depth) and to the input
  /// depth map (input_depth). All depths are quantized to bytes (uchar) before comparison, and a
  /// depth value is considered to match the ground truth if the absolute difference between its
  /// quantized depth and the LIDAR point's is below 'delta_max'. Based on the evaluation method
  /// from [0].
  ///
  /// TODO-LOW(andrei): Honestly, the [0] approach is not very good. It would make more sense to use
  /// float metric depth, and make the deltas also metric, i.e., more meaningful...
  ///
  /// [0]: Sengupta, S., Greveson, E., Shahrokni, A., & Torr, P. H. S. (2013). Urban 3D semantic modelling using stereo vision. Proceedings - IEEE International Conference on Robotics and Automation, 580–585. https://doi.org/10.1109/ICRA.2013.6630632
  DepthEvaluation EvaluateDepth(
      const DepthEvaluationMeta &meta,
      const Eigen::MatrixX4f &lidar_points,
      const uchar *const rendered_depth,
      const uchar *const input_depth,
      const Eigen::Matrix4f &velo_to_cam,
      const Eigen::MatrixXf &cam_proj,
      const int frame_width,
      const int frame_height,
      const float min_depth_meters,
      const float max_depth_meters,
      const uint delta_max,
      const uint rendered_stride = 4,
      const uint input_stride = 4
  ) const;

  virtual ~Evaluation() {
    delete velodyne;
  }

  Velodyne *GetVelodyne() {
    return velodyne;
  }

  const Velodyne *GetVelodyne() const {
    return velodyne;
  }

 private:
  Velodyne *velodyne;

};

}
}

#endif //DYNSLAM_EVALUATION_H
