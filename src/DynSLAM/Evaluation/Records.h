// Record types used for logging stats.

#ifndef DYNSLAM_RECORDS_H
#define DYNSLAM_RECORDS_H

#include <string>

#include "CsvWriter.h"

namespace dynslam {
namespace eval {

struct DepthResult : public ICsvSerializable {
  const long measurement_count;
  const long error_count;
  const long missing_count;
  const long correct_count;
  const long missing_separate_count;

  DepthResult(const long measurement_count,
              const long error_count,
              const long missing_count,
              const long correct_count,
              const long missing_separate_count)
      : measurement_count(measurement_count),
        error_count(error_count),
        missing_count(missing_count),
        correct_count(correct_count),
        missing_separate_count(missing_separate_count)
  {
    assert(measurement_count == (error_count + missing_count + correct_count));
    // 'missing_count' may be more generous, and count a pixel as missing if it's missing in EITHER
    // the fusion or the input.
    assert(missing_count >= missing_separate_count);
  }

  /// \brief Returns the ratio of correct pixels in this depth evaluation result.
  /// \param include_missing Whether to count pixels with no depth data in the evaluated depth map
  ///                        as incorrect.
  double GetCorrectPixelRatio(bool include_missing) const {
    if (include_missing) {
      return static_cast<double>(correct_count) / measurement_count;
    } else {
      return static_cast<double>(correct_count) / (measurement_count - missing_count);
    }
  }

  std::string GetHeader() const override {
    return "measurements_count,error_count,missing_count,correct_count,missing_separate_count";
  }

  std::string GetData() const override {
    return utils::Format("%d,%d,%d,%d,%d",
                         measurement_count,
                         error_count,
                         missing_count,
                         correct_count,
                         missing_separate_count);
  }
};

struct DepthEvaluationMeta {
  const int frame_idx;
  const std::string dataset_id;

  DepthEvaluationMeta(const int frame_idx, const std::string &dataset_id)
      : frame_idx(frame_idx), dataset_id(dataset_id) {}
};

/// \brief Stores the result of comparing a computed depth with a LIDAR ground truth.
struct DepthEvaluation : public ICsvSerializable {
  const float delta_max;
  /// \brief Results for the depth map synthesized from engine.
  const DepthResult fused_result;
  /// \brief Results for the depth map received as input.
  const DepthResult input_result;
  /// \brief Whether errors were computed in the KITTI stereo benchmark (2015) style, i.e.,
  ///        error if (delta>delta_max AND delta>5%GT). Otherwise, the '>5%GT' condition is not
  ///        used.
  const bool kitti_style;

  DepthEvaluation(const float delta_max,
                  DepthResult &&fused_result,
                  DepthResult &&input_result,
                  bool kitti_style)
      : delta_max(delta_max),
        fused_result(fused_result),
        input_result(input_result),
        kitti_style(kitti_style)
  {}

  std::string GetHeader() const override {
    const std::string kitti_label = (kitti_style ? "-kitti" : "");
    return utils::Format(
        "fusion-total-%.2f%s,fusion-error-%.2f%s,fusion-missing-%.2f%s,fusion-correct-%.2f%s,fusion-missing-separate-%.2f%s,"
        "input-total-%.2f%s,input-error-%.2f%s,input-missing-%.2f%s,input-correct-%.2f%s,input-missing-separate-%.2f%s",
        delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
        delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
        delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
        delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
        delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str());
  }

  std::string GetData() const override {
    return utils::Format("%s,%s", fused_result.GetData().c_str(), input_result.GetData().c_str());
  }
};

/// \brief Contains a frame's depth evaluation results for multiple values of $\delta_max$.
struct DepthFrameEvaluation : public ICsvSerializable {
  const DepthEvaluationMeta meta;
  const float max_depth_meters;
  const std::vector<DepthEvaluation> evaluations;

  DepthFrameEvaluation(DepthEvaluationMeta &meta,
                       float max_depth_meters,
                       std::vector<DepthEvaluation> &&evaluations)
      : meta(meta), max_depth_meters(max_depth_meters), evaluations(evaluations) {}

  std::string GetHeader() const override {
    std::stringstream ss;
    ss << "frame";
    for (auto &eval : evaluations) {
      ss << "," << eval.GetHeader();
    }
    return ss.str();
  }

  std::string GetData() const override {
    std::stringstream ss;
    ss << meta.frame_idx;
    for (auto &eval :evaluations) {
      ss << "," << eval.GetData();
    }
    return ss.str();
  }
};

/// \brief The evaluation of a single pose at a single time.
struct TrackletEvaluation : public ICsvSerializable {
  int frame_id;
  int track_id;
  double trans_error;
  double rot_error;

  TrackletEvaluation(int frame_id, int track_id, double trans_error, double rot_error)
      : frame_id(frame_id),
        track_id(track_id),
        trans_error(trans_error),
        rot_error(rot_error) {}

  std::string GetHeader() const override {
    return "frame_id,track_id,trans_error,rot_error";
  }

  std::string GetData() const override {
    return utils::Format("%d,%d,%lf,%lf", frame_id, track_id, trans_error, rot_error);
  }
};

// Only tracks the static map memory usage, since that dominates everything else anyway.
struct MemoryUsageEntry : public ICsvSerializable {
  int frame_id;
  size_t memory_usage_bytes;
  size_t saved_memory_cum_bytes;
  // Very wasteful and lazy, since they don't change.
  VoxelDecayParams decay_params;

  MemoryUsageEntry(int frame_id,
                   size_t memory_usage_mb,
                   size_t saved_memory_cum_mb,
                   const VoxelDecayParams &decay_params)
      : frame_id(frame_id),
        memory_usage_bytes(memory_usage_mb),
        saved_memory_cum_bytes(saved_memory_cum_mb),
        decay_params(decay_params) {}

  std::string GetHeader() const override {
    return "frame_id,memory_usage_bytes,saved_memory_cum_bytes,decay_enabled,decay_min_age,decay_max_weight";
  }

  std::string GetData() const override {
    return utils::Format("%d,%zu,%zu,%d,%d,%d",
                         frame_id,
                         memory_usage_bytes,
                         saved_memory_cum_bytes,
                         static_cast<int>(decay_params.enabled),
                         decay_params.min_decay_age,
                         decay_params.max_decay_weight);
  }
};

}
}

#endif //DYNSLAM_RECORDS_H
