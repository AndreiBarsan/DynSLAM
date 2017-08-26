
#ifndef DYNSLAM_EVALUATION_H
#define DYNSLAM_EVALUATION_H

#include "../DynSlam.h"
#include "../Input.h"

#include "CsvWriter.h"
#include "ILidarEvalCallback.h"
#include "Records.h"
#include "Tracklets.h"
#include "VelodyneIO.h"

// TODO-LOW(andrei): Don't directly rely on flags. Put the flags in a proper config and rely on that
// for naming things, generating metadata, etc.
DECLARE_int32(max_decay_weight);
DECLARE_int32(fusion_every);

namespace dynslam {
class DynSlam;
}

namespace dynslam {
namespace eval {

// Used internally as an accumulator.
struct Stats {
  long missing = 0;
  long error = 0;
  long correct = 0;
  // Keeps track of the # of missing values even when 'compare_on_intersection' is true.
  long missing_separate = 0;
};



/// \brief Main class handling the quantitative evaluation of the DynSLAM system.
///
/// A disparity is counted as accurate if the absoluted difference between it and the ground truth
/// disparity is less than 'delta_max'. Based on the evaluation method from [0].
///
/// We also perform a KITTI-Stereo-2015-style evaluation, counting the number of pixels whose
/// disparity exceeds 3px AND 5% of the ground truth value.
///
/// Evaluation end error visualizations are computed using a callback structure: the method
/// 'EvaluateDepth' goes through a frame's LIDAR pointcloud and projects its points onto the left
/// camera frame, discarding invalid points which fall outside it. For every point with valid
/// ground truth (LIDAR), input, and fused depth, the system invokes a list of callbacks, which do
/// things like error visualization, error aggregation, etc.
///
/// [0]: Sengupta, S., Greveson, E., Shahrokni, A., & Torr, P. H. S. (2013). Urban 3D semantic modelling using stereo vision. Proceedings - IEEE International Conference on Robotics and Automation, 580–585. https://doi.org/10.1109/ICRA.2013.6630632
class Evaluation {
 public:
  // TODO(andrei): We could reduce code duplication in these helpers by e.g., grouping the misc
  // config parameters into a config object.
  static std::string GetBaseCsvName(
      const std::string &dataset_root,
      const Input *input,
      float voxel_size_meters,
      bool direct_refinement,
      bool is_dynamic,
      bool use_depth_weighting,
      const string &base_folder = "../csv"
  ) {
    return utils::Format(
        "%s/k-%d-%s-offset-%d-depth-%s-voxelsize-%.4f-max-depth-m-%.2f-%s-%s-%s%s",
        base_folder.c_str(),
        FLAGS_max_decay_weight,
        input->GetDatasetIdentifier().c_str(),
        input->GetCurrentFrame(),
        input->GetDepthProvider()->GetName().c_str(),
        voxel_size_meters,
        input->GetDepthProvider()->GetMaxDepthMeters(),
        is_dynamic ? "dynamic-mode" : "NO-dynamic",
        direct_refinement ? "with-direct-ref" : "NO-direct-ref",
        use_depth_weighting ? "with-fusion-weights" : "NO-fusion-weights",
        FLAGS_fusion_every == 1 ? "" : utils::Format("-fuse-every-%d", FLAGS_fusion_every).c_str()
    );
  }

  static std::string GetDepthCsvName(const std::string &dataset_root,
                                     const Input *input,
                                     float voxel_size_meters,
                                     bool direct_refinement,
                                     bool is_dynamic,
                                     bool use_depth_weighting
  ) {
    return utils::Format("%s-unified-depth-result.csv",
                         GetBaseCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting).c_str());
  }

  static std::string GetDynamicDepthCsvName(const std::string &dataset_root,
                                            const Input *input,
                                            float voxel_size_meters,
                                            bool direct_refinement,
                                            bool is_dynamic,
                                            bool use_depth_weighting
  ) {
    return utils::Format("%s-dynamic-depth-result.csv",
                         GetBaseCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting).c_str());
  }

  static std::string GetStaticDepthCsvName(const std::string &dataset_root,
                                            const Input *input,
                                            float voxel_size_meters,
                                            bool direct_refinement,
                                            bool is_dynamic,
                                            bool use_depth_weighting
  ) {
    return utils::Format("%s-static-depth-result.csv",
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

  static std::string GetMemoryCsvName(const std::string &dataset_root,
                                      const Input *input,
                                      float voxel_size_meters,
                                      bool direct_refinement,
                                      bool is_dynamic,
                                      bool use_depth_weighting
  ) {
    return utils::Format("%s-memory.csv",
                         GetBaseCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting).c_str());
  }

 public:
  const Eigen::Matrix4d velo_to_left_gray_cam_;
  const Eigen::Matrix34d proj_left_color_;
  const Eigen::Matrix34d proj_right_color_;
  const float baseline_m_;
  const int frame_width_;
  const int frame_height_;
  const float min_depth_m_;
  const float max_depth_m_;
  const float left_focal_length_px_;

  Evaluation(const std::string &dataset_root,
             const Input *input,
             const Eigen::Matrix4d &velo_to_left_gray_cam,
             const Eigen::Matrix34d &proj_left_color,
             const Eigen::Matrix34d &proj_right_color,
             const float baseline_m,
             const int frame_width,
             const int frame_height,
             float voxel_size_meters,
             bool direct_refinement,
             bool is_dynamic,
             bool use_depth_weighting,
             bool separate_static_and_dynamic)
      : velo_to_left_gray_cam_(velo_to_left_gray_cam),
        proj_left_color_(proj_left_color),
        proj_right_color_(proj_right_color),
        baseline_m_(baseline_m),
        frame_width_(frame_width),
        frame_height_(frame_height),
        min_depth_m_(input->GetDepthProvider()->GetMinDepthMeters()),
        max_depth_m_(input->GetDepthProvider()->GetMaxDepthMeters()),
        left_focal_length_px_(static_cast<float>(proj_left_color_(0, 0))),
        velodyne_(new VelodyneIO(utils::Format("%s/%s",
                                             dataset_root.c_str(),
                                             input->GetConfig().velodyne_folder.c_str()),
                                input->GetConfig().velodyne_fname_format)),
        csv_unified_depth_dump_(GetDepthCsvName(dataset_root, input, voxel_size_meters, direct_refinement,
                                        is_dynamic, use_depth_weighting)),
        csv_tracking_dump_(GetTrackingCsvName(dataset_root, input, voxel_size_meters,
                                              direct_refinement, is_dynamic, use_depth_weighting)),
        separate_static_and_dynamic_(separate_static_and_dynamic),
        csv_static_depth_dump_(GetStaticDepthCsvName(dataset_root, input, voxel_size_meters,
                                                     direct_refinement, is_dynamic,
                                                     use_depth_weighting)),
        csv_dynamic_depth_dump_(GetDynamicDepthCsvName(dataset_root, input, voxel_size_meters,
                                                       direct_refinement, is_dynamic,
                                                       use_depth_weighting)),
        csv_memory_(GetMemoryCsvName(dataset_root, input, voxel_size_meters,
                                                       direct_refinement, is_dynamic,
                                                       use_depth_weighting)),
        eval_tracklets_(! input->GetConfig().tracklet_folder.empty())
  {
    if (this->eval_tracklets_) {
      std::string tracklet_fpath = utils::Format("%s/%s", dataset_root.c_str(),
                                                 input->GetConfig().tracklet_folder.c_str());
      frame_to_tracklets_ = ReadGroupedTracklets(tracklet_fpath);
      cout << "[Evaluation] Evaluating tracklets." << endl;
    }
    else {
      cout << "[Evaluation] Not evaluating tracklets." << endl;
    }

    if (is_dynamic) {
      cout << "[Evaluation] DYNAMIC MODE IS ON FOR LOGGING" << endl;
    }
    else {
      cout << "[Evaluation] DYNAMIC MODE IS OFF" << endl;
    }

    cout << "Dumping depth data to file: " << csv_unified_depth_dump_.output_fpath_ << endl;
  }

  Evaluation(const Evaluation&) = delete;
  Evaluation(const Evaluation&&) = delete;
  Evaluation& operator=(const Evaluation&) = delete;
  Evaluation& operator=(const Evaluation&&) = delete;

  virtual ~Evaluation() {
    delete velodyne_;
  }

  /// \brief Supermethod in charge of all per-frame evaluation metrics except memory.
  void EvaluateFrame(Input *input,
                     DynSlam *dyn_slam,
                     int frame_idx,
                     bool enable_compositing);

  void LogMemoryUse(const DynSlam *dyn_slam) {
    MemoryUsageEntry memory_usage(
        dyn_slam->GetCurrentFrameNo() - 1,
        dyn_slam->GetStaticMapMemoryBytes(),
        dyn_slam->GetStaticMapSavedDecayMemoryBytes(),
        dyn_slam->GetStaticMapDecayParams()
    );

    csv_memory_.Write(memory_usage);
  }

  /// hacky extension for separate eval (static+dynamic)
  std::pair<DepthFrameEvaluation,
          DepthFrameEvaluation> EvaluateFrameSeparate(int dynslam_frame_idx,
                                                      bool enable_compositing,
                                                      Input *input,
                                                      DynSlam *dyn_slam);

  DepthFrameEvaluation EvaluateFrame(int frame_idx,
                                       bool enable_compositing,
                                       Input *input,
                                       DynSlam *dyn_slam);

  static uint depth_delta(uchar computed_depth, uchar ground_truth_depth) {
    return static_cast<uint>(abs(
        static_cast<int>(computed_depth) - static_cast<int>(ground_truth_depth)));
  }

  /// \brief Compares a fused depth map and input depth map to the corresponding LIDAR pointcloud,
  ///        which is considered to be the ground truth.
  /// \note This method does not compute any metrics on its own. It relies on callbacks to do this.
  ///
  /// Projects each LIDAR point into both the left and the right camera frames, in order to compute
  /// the ground truth disparity. Then, if input and/or rendered depth values are available at
  /// those coordinates, computes their corresponding disparity as well, comparing it to the ground
  /// truth disparity. These values are then passed to a list of possible callbacks which can be
  /// tasked with, e.g., visualization, accuracy computations, etc.
  void EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                     const float *const rendered_depth,
                     const cv::Mat1s &input_depth_mm,
                     const std::vector<ILidarEvalCallback *> &callbacks) const;

  /// \brief Simplistic evaluation of tracking performance, mostly meant to asses whether using the
  ///        direct refinement steps leads to any improvement.
  vector<TrackletEvaluation> EvaluateTracking(Input *input, DynSlam *dyn_slam);

  VelodyneIO *GetVelodyneIO() {
    return velodyne_;
  }

  const VelodyneIO *GetVelodyneIO() const {
    return velodyne_;
  }

  SUPPORT_EIGEN_FIELDS;

 protected:
  bool ProjectLidar(const Eigen::Vector4f &velodyne_reading,
                    Eigen::Vector3d& out_velo_2d_left,
                    Eigen::Vector3d& out_velo_2d_right) const;

 private:
  VelodyneIO *velodyne_;
  // CSV results are written here when static and dynamic parts are NOT evaluated separately.
  CsvWriter csv_unified_depth_dump_;
  CsvWriter csv_tracking_dump_;

  bool separate_static_and_dynamic_;
  CsvWriter csv_static_depth_dump_;
  CsvWriter csv_dynamic_depth_dump_;
  CsvWriter csv_memory_;

  const bool eval_tracklets_;
  std::map<int, std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>> frame_to_tracklets_;
};

}
}

#endif //DYNSLAM_EVALUATION_H
