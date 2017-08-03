#ifndef DYNSLAM_DYNSLAM_H
#define DYNSLAM_DYNSLAM_H

#include <Eigen/StdVector>

#include <pangolin/display/opengl_render_state.h>

#include "InfiniTamDriver.h"
#include "InstRecLib/InstanceReconstructor.h"
#include "InstRecLib/PrecomputedSegmentationProvider.h"
#include "InstRecLib/SparseSFProvider.h"
#include "Input.h"

namespace dynslam {
namespace eval {
class Evaluation;
}
}

namespace dynslam {

using namespace instreclib;
using namespace instreclib::reconstruction;
using namespace instreclib::segmentation;
using namespace dynslam::drivers;

// TODO(andrei): Get rid of ITM-specific image objects for visualization.
/// \brief The central class of the DynSLAM system.
/// It processes input stereo frames and generates separate maps for all encountered object
/// instances, as well one for the static background.
class DynSlam {
 public:
  DynSlam(InfiniTamDriver *itm_static_scene_engine,
          SegmentationProvider *segmentation_provider,
          SparseSFProvider *sparse_sf_provider,
          dynslam::eval::Evaluation *evaluation,
          const Vector2i &input_shape,
          float max_depth_meters,
          const Eigen::Matrix34f& proj_left_rgb,
          const Eigen::Matrix34f& proj_right_rgb)
    : static_scene_(itm_static_scene_engine),
      segmentation_provider_(segmentation_provider),
      instance_reconstructor_(new InstanceReconstructor(itm_static_scene_engine)),
      sparse_sf_provider_(sparse_sf_provider),
      evaluation_(evaluation),
      // Allocate the ITM buffers on the CPU and on the GPU (true, true).
      out_image_(new ITMUChar4Image(input_shape, true, true)),
      out_image_float_(new ITMFloatImage(input_shape, true, true)),
      input_rgb_image_(new cv::Mat3b(input_shape.x, input_shape.y)),
      input_raw_depth_image_(new cv::Mat1s(input_shape.x, input_shape.y)),
      current_frame_no_(0),
      input_width_(input_shape.x),
      input_height_(input_shape.y),
      max_depth_meters_(max_depth_meters),
      poses_({ Eigen::Matrix4f::Identity() }),
      projection_matrix_left_rgb_(proj_left_rgb),
      projection_matrix_right_rgb_(proj_right_rgb)
  {}

  /// \brief Reads in and processes the next frame from the data source.
  /// This is where most of the interesting stuff happens.
  void ProcessFrame(Input *input);

  /// \brief Returns an RGB preview of the latest color frame.
  const cv::Mat3b* GetRgbPreview() {
    return input_rgb_image_;
  }

  /// \brief Returns an RGB preview of the static parts from the latest color frame.
  const cv::Mat3b* GetStaticRgbPreview() {
    return static_scene_->GetRgbPreview();
  }

  /// \brief Returns a preview of the latest depth frame.
  const cv::Mat1s* GetDepthPreview() {
    return input_raw_depth_image_;
  }

  /// \brief Returns a preview of the static parts from the latest depth frame.
  const cv::Mat1s* GetStaticDepthPreview() {
    return static_scene_->GetDepthPreview();
  }

  /// \brief Returns a preview of the reconstructed object indicated by `object_idx`.
  const unsigned char* GetObjectRaycastPreview(
      int object_idx,
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview
  ) {
    instance_reconstructor_->GetInstanceRaycastPreview(out_image_, object_idx, model_view, preview);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  /// \brief Returns an RGBA preview of the reconstructed static map.
  const unsigned char* GetStaticMapRaycastPreview(
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview
  ) {
    static_scene_->GetVisualizationEngine()->SetMaxDepthMeters(max_depth_meters_);
    static_scene_->GetImage(out_image_, preview, model_view);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  const float* GetStaticMapRaycastDepthPreview(const pangolin::OpenGlMatrix &model_view) {
    static_scene_->GetFloatImage(out_image_float_, PreviewType::kDepth, model_view);
    return out_image_float_->GetData(MEMORYDEVICE_CPU);
  }

  /// \brief Returns an RGBA unsigned char frame containing the preview of the most recent frame's
  /// semantic segmentation.
  const cv::Mat3b* GetSegmentationPreview() {
    if (segmentation_provider_->GetSegResult() == nullptr) {
      return nullptr;
    }

    return segmentation_provider_->GetSegResult();
  }

  /// \brief Returns an **RGBA** preview of the latest segmented object instance.
  const unsigned char* GetObjectPreview(int object_idx);

  const float* GetObjectDepthPreview(int object_idx) {
    ITMFloatImage *preview = instance_reconstructor_->GetInstancePreviewDepth(static_cast<size_t>(object_idx));

    if (nullptr == preview) {
      // This happens when there's no instances to preview.
      out_image_float_->Clear();
    } else {
      out_image_float_->SetFrom(preview, ORUtils::MemoryBlock<float>::CPU_TO_CPU);
    }

    return out_image_float_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  }

  InstanceReconstructor* GetInstanceReconstructor() {
    return instance_reconstructor_;
  }

  dynslam::eval::Evaluation* GetEvaluation() {
    return evaluation_;
  }

  int GetInputWidth() const {
    return input_width_;
  }

  int GetInputHeight() const {
    return input_height_;
  }

  int GetCurrentFrameNo() const {
    return current_frame_no_;
  }

  void SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const;

  void ForceDynamicObjectCleanup(int object_id) {
    instance_reconstructor_->ForceObjectCleanup(object_id);
  }

  void SaveDynamicObject(const std::string &dataset_name, const std::string &depth_name, int object_id) const;

  // Variants would solve this nicely, but they are C++17-only... TODO(andrei): Use Option<>.
  // Will error out if no flow information is available.
  const SparseSceneFlow& GetLatestFlow() {
    return sparse_sf_provider_->GetFlow();
  }

  /// \brief Returns the most recent egomotion computed by the primary tracker.
  /// Composing these transforms from the first frame is equivalent to the `GetPose()` method, which
  /// returns the absolute current pose.
  Eigen::Matrix4f GetLastEgomotion() const {
    return static_scene_->GetLastEgomotion();
  }

  /// \brief Returns the current pose of the camera in the coordinate frame used by the tracker.
  /// For the KITTI dataset (and the KITTI-odometry one) this represents the center of the left
  /// camera.
  Eigen::Matrix4f GetPose() const {
    return static_scene_->GetPose();
  }

  // TODO(andrei): Similar stats also taking instances (and maybe views) into account. Take into
  // account the destroyed tracks (old ones) somehow.
  size_t GetStaticMapMemory() const {
    return static_scene_->GetUsedMemoryBytes();
  }

  size_t GetStaticMapSavedDecayMemory() const {
    return static_scene_->GetSavedDecayMemory();
  }

  void WaitForJobs() {
    static_scene_->WaitForMeshDump();
  }

  const Eigen::Matrix34f GetLeftRgbProjectionMatrix() const {
    return projection_matrix_left_rgb_;
  };

  const Eigen::Matrix34f GetRightRgbProjectionMatrix() const {
    return projection_matrix_right_rgb_;
  };

  SUPPORT_EIGEN_FIELDS;

private:
  InfiniTamDriver *static_scene_;
  SegmentationProvider *segmentation_provider_;
  InstanceReconstructor *instance_reconstructor_;
  SparseSFProvider *sparse_sf_provider_;
  dynslam::eval::Evaluation *evaluation_;

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;

  int current_frame_no_;
  int input_width_;
  int input_height_;

  /// Whether to force instance reconstruction even for non-dynamic objects, like parked cars.
  bool always_reconstruct_objects_ = true;

  float max_depth_meters_;

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_;

  /// \brief Matrix for projecting 3D homogeneous coordinates in the camera's coordinate frame to
  ///       2D homogeneous coordinates expressed in pixels.
  const Eigen::Matrix34f projection_matrix_left_rgb_;
  const Eigen::Matrix34f projection_matrix_right_rgb_;

  /// \brief Returns a path to the folder where the dataset's meshes should be dumped, creating it
  ///        using a naive system call if it does not exist.
  std::string EnsureDumpFolderExists(const string& dataset_name) const;

};

}

#endif //DYNSLAM_DYNSLAM_H
