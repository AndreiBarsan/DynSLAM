
#ifndef DYNSLAM_EVALUATION_H
#define DYNSLAM_EVALUATION_H

#include "../DynSlam.h"
#include "../Input.h"

#include "CsvWriter.h"
#include "ILidarEvalCallback.h"
#include "Tracklets.h"
#include "Velodyne.h"

namespace dynslam {
class DynSlam;
}

namespace dynslam {
namespace eval {

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
  ///                        as incorrect.
  double GetCorrectPixelRatio(bool include_missing) const {
    if (include_missing) {
      return static_cast<double>(correct_count) / measurement_count;
    } else {
      return static_cast<double>(correct_count) / (measurement_count - missing_count);
    }
  }

  string GetHeader() const override {
    return "measurements_count,error_count,missing_count,correct_count";
  }

  string GetData() const override {
    return utils::Format("%d,%d,%d,%d",
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
struct DepthEvaluation : public ICsvSerializable {
  const float delta_max;
  /// \brief Results for the depth map synthesized from engine.
  const DepthResult fused_result;
  /// \brief Results for the depth map received as input.
  const DepthResult input_result;
  /// \brief Whether errors were computed in the KITTI stereo benchmark (2015) style, i.e.,
  ///        error if (delta>delta_max AND delta>5%GT).
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

  string GetHeader() const override {
    const string kitti_label = (kitti_style ? "-kitti" : "");
    return utils::Format("fusion-total-%.2f%s,fusion-error-%.2f%s,fusion-missing-%.2f%s,fusion-correct-%.2f%s,"
                         "input-total-%.2f%s,input-error-%.2f%s,input-missing-%.2f%s,input-correct-%.2f%s",
                         delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
                         delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
                         delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str(),
                         delta_max, kitti_label.c_str(), delta_max, kitti_label.c_str());
  }

  string GetData() const override {
    return utils::Format("%s,%s", fused_result.GetData().c_str(), input_result.GetData().c_str());
  }
};

/// \brief Contains a frame's depth evaluation results for multiple values of $\delta_max$.
struct DepthFrameEvaluation : public ICsvSerializable {
  const DepthEvaluationMeta meta;
  const float max_depth_meters;
  const std::vector<DepthEvaluation> evaluations;

  DepthFrameEvaluation(DepthEvaluationMeta &&meta,
                       float max_depth_meters,
                       vector<DepthEvaluation> &&evaluations)
      : meta(meta), max_depth_meters(max_depth_meters), evaluations(evaluations) {}

  string GetHeader() const override {
    std::stringstream ss;
    ss << "frame";
    for (auto &eval : evaluations) {
      ss << "," << eval.GetHeader();
    }
    return ss.str();
  }

  string GetData() const override {
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

  string GetHeader() const override {
    return "frame_id,track_id,trans_error,rot_error";
  }

  string GetData() const override {
    return utils::Format("%d,%d,%lf,%lf", frame_id, track_id, trans_error, rot_error);
  }
};

/// \brief Main class handling the quantitative evaluation of the DynSLAM system.
class Evaluation {
 public:

  static std::string GetBaseCsvName(
      const std::string &dataset_root,
      const Input *input,
      float voxel_size_meters,
      bool direct_refinement,
      bool is_dynamic,
      bool use_depth_weighting,
      const string &base_folder = "../csv"
  ) {
    if (is_dynamic) {
      cout << "[Evaluation] DYNAMIC MODE IS ON FOR LOGGING" << endl;
    }
    else {
      cout << "[Evaluation] DYNAMIC MODE IS OFF" << endl;
    }
    return utils::Format("%s/%s-offset-%d-depth-%s-voxelsize-%.4f-max-depth-m-%.2f-%s-%s-%s",
                         base_folder.c_str(),
                         input->GetDatasetIdentifier().c_str(),
                         input->GetCurrentFrame(),
                         input->GetDepthProvider()->GetName().c_str(),
                         voxel_size_meters,
                         input->GetDepthProvider()->GetMaxDepthMeters(),
                         is_dynamic ? "dynamic-mode" : "NO-dynamic",
                         direct_refinement ? "with-direct-ref" : "NO-direct-ref",
                         use_depth_weighting ? "with-fusion-weights" : "NO-fusion-weights");
  }

  static std::string GetDepthCsvName(const std::string &dataset_root,
                                     const Input *input,
                                     float voxel_size_meters,
                                     bool direct_refinement,
                                     bool is_dynamic,
                                     bool use_depth_weighting
  ) {
    return utils::Format("%s-depth-result.csv",
                         GetBaseCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting).c_str());
  }

  static std::string GetTrackingCsvName(const std::string &dataset_root,
                                        const Input *input,
                                        float voxel_size_meters,
                                        bool direct_refinement,
                                        bool is_dynamic,
                                        bool use_depth_weighting
  ) {
    return utils::Format("%s-3d-tracking-result.csv",
                         GetBaseCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting).c_str());
  }

 public:
  Evaluation(const std::string &dataset_root,
             const Input *input,
             const Eigen::Matrix4d &velodyne_to_rgb,
             float voxel_size_meters,
             bool direct_refinement,
             bool is_dynamic,
             bool use_depth_weighting)
      : velodyne_(new Velodyne(utils::Format("%s/%s",
                                             dataset_root.c_str(),
                                             input->GetConfig().velodyne_folder.c_str()),
                                input->GetConfig().velodyne_fname_format,
                                velodyne_to_rgb)),
        csv_depth_dump_(GetDepthCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting)),
        csv_tracking_dump_(GetTrackingCsvName(dataset_root, input, voxel_size_meters,
                                              direct_refinement, is_dynamic, use_depth_weighting)),
        eval_tracklets_(! input->GetConfig().tracklet_folder.empty())
  {
    if (this->eval_tracklets_) {
//      cout << "Found tracklet GT data. Enabling track evaluation!" << endl;
      std::string tracklet_fpath = utils::Format("%s/%s", dataset_root.c_str(),
                                                 input->GetConfig().tracklet_folder.c_str());
      frame_to_tracklets_ = ReadGroupedTracklets(tracklet_fpath);
    }

    cout << "Dumping depth data to file: " << csv_depth_dump_.output_fpath_ << endl;
  }

  Evaluation(const Evaluation&) = delete;
  Evaluation(const Evaluation&&) = delete;
  Evaluation& operator=(const Evaluation&) = delete;
  Evaluation& operator=(const Evaluation&&) = delete;

  virtual ~Evaluation() {
    delete velodyne_;
  }

  /// \brief Supermethod in charge of all per-frame evaluation metrics.
  void EvaluateFrame(Input *input, DynSlam *dyn_slam);

  DepthFrameEvaluation EvaluateFrame(int frame_idx, Input *input, DynSlam *dyn_slam);

  static uint depth_delta(uchar computed_depth, uchar ground_truth_depth) {
    return static_cast<uint>(abs(
        static_cast<int>(computed_depth) - static_cast<int>(ground_truth_depth)));
  }

  /// \brief Compares a fused depth map and input depth map to the corresponding LIDAR pointcloud,
  ///        which is considered to be the ground truth.
  /// \param compare_on_intersection If true, then the accuracy of both input and fused depth is
  /// computed only for ground truth LIDAR points which have both corresponding input depth, as well
  /// as fused depth. Otherwise, the input and depth accuracies are compute separately.
  ///
  /// Projects each LIDAR point into both the left and the right camera frames, in order to compute
  /// the ground truth disparity. Then, if input and/or rendered depth values are available at
  /// those coordinates, computes their corresponding disparity as well, comparing it to the ground
  /// truth disparity.
  ///
  /// A disparity is counted as accurate if the absoluted difference between it and the ground truth
  /// disparity is less than 'delta_max'. Based on the evaluation method from [0].
  ///
  /// We also perform a KITTI-Stereo-2015-style evaluation, counting the number of pixels whose
  /// disparity exceeds 3px AND 5% of the ground truth value.
  ///
  /// [0]: Sengupta, S., Greveson, E., Shahrokni, A., & Torr, P. H. S. (2013). Urban 3D semantic modelling using stereo vision. Proceedings - IEEE International Conference on Robotics and Automation, 580–585. https://doi.org/10.1109/ICRA.2013.6630632
  DepthEvaluation EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                const float *const rendered_depth,
                                const cv::Mat1s &input_depth_mm,
                                const Eigen::Matrix4d &velo_to_left_gray_cam,
                                const Eigen::Matrix34d &proj_left_color,
                                const Eigen::Matrix34d &proj_right_color,
                                const float baseline_m,
                                const int frame_width,
                                const int frame_height,
                                const float min_depth_meters,
                                const float max_depth_meters,
                                const float delta_max,
                                const bool compare_on_intersection,
                                bool kitti_style,
                                ILidarEvalCallback *callback) const;

  /// \brief Simplistic evaluation of tracking performance, mostly meant to asses whether using the
  ///        direct refinement steps leads to any improvement.
  vector<TrackletEvaluation> EvaluateTracking(Input *input, DynSlam *dyn_slam);

  Velodyne *GetVelodyne() {
    return velodyne_;
  }

  const Velodyne *GetVelodyne() const {
    return velodyne_;
  }

  SUPPORT_EIGEN_FIELDS;

 private:
  Velodyne *velodyne_;
  CsvWriter csv_depth_dump_;
  CsvWriter csv_tracking_dump_;

  const bool eval_tracklets_;
  std::map<int, std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>> frame_to_tracklets_;
};

}
}

#endif //DYNSLAM_EVALUATION_H
