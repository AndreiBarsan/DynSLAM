#ifndef DYNSLAM_DYNSLAM_H
#define DYNSLAM_DYNSLAM_H

#include <pangolin/display/opengl_render_state.h>

#include "InfiniTamDriver.h"
#include "InstRecLib/InstanceReconstructor.h"
#include "InstRecLib/PrecomputedSegmentationProvider.h"
#include "InstRecLib/SparseSFProvider.h"
#include "Input.h"
#include "Evaluation/Evaluation.h"

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
  // TODO(andrei): If possible, get rid of the initialize method.
  void Initialize(InfiniTamDriver *itm_static_scene_engine,
                  SegmentationProvider *segmentation_provider,
                  SparseSFProvider *sparse_sf_provider,
                  dynslam::eval::Evaluation *evaluation);

  /// \brief Reads in and processes the next frame from the data source.
  /// This is where most of the interesting stuff happens.
  void ProcessFrame(Input *input);

  const unsigned char* GetRaycastPreview() {
    static_scene_->GetImage(out_image_, PreviewType::kLatestRaycast);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

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

  /// \brief Returns a preview of the reconstructed static map.
  const unsigned char* GetStaticMapRaycastPreview(
      int object_idx,
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview
  ) {
    static_scene_->GetImage(out_image_, preview, model_view);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
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

  void SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const {
    string target_folder = EnsureDumpFolderExists(dataset_name);
    string map_fpath = utils::Format("%s/static-%s-mesh-%06d-frames.obj",
                                     target_folder.c_str(),
                                     depth_name.c_str(),
                                     current_frame_no_);
    cout << "Saving full static map to: " << map_fpath << endl;
    static_scene_->SaveSceneToMesh(map_fpath.c_str());
  }

  void SaveDynamicObject(const std::string &dataset_name, const std::string &depth_name, int object_id) const {
    cout << "Saving mesh for object #" << object_id << "'s reconstruction..." << endl;
    string target_folder = EnsureDumpFolderExists(dataset_name);
    string instance_fpath = utils::Format("%s/instance-%s-%06d-mesh.obj",
                                          target_folder.c_str(),
                                          depth_name.c_str(),
                                          object_id);
    instance_reconstructor_->SaveObjectToMesh(object_id, instance_fpath);

    cout << "Done saving mesh for object #" << object_id << "'s reconstruction in file ["
         << instance_fpath << "]." << endl;
  }

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

private:
  InfiniTamDriver *static_scene_;
  SegmentationProvider *segmentation_provider_;
  InstanceReconstructor *instance_reconstructor_;
  SparseSFProvider *sparse_sf_provider_;
  dynslam::eval::Evaluation *evaluation_;

  ITMUChar4Image *out_image_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;

  int current_frame_no_;
  int input_width_;
  int input_height_;

  /// Whether to force instance reconstruction even for static ones.
  bool always_reconstruct_objects_ = false;

  bool enable_map_decay_ = true;

  /// \brief Returns a path to the folder where the dataset's meshes should be dumped, creating it
  ///        using a naive system call if it does not exist.
  std::string EnsureDumpFolderExists(const string& dataset_name) const {
    // TODO-LOW(andrei): Make this more cross-platform and more secure.
    string today_folder = utils::GetDate();
    string target_folder = "mesh_out/" + dataset_name + "/" + today_folder;
    if(system(utils::Format("mkdir -p '%s'", target_folder.c_str()).c_str())) {
      throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
    }

    return target_folder;
  }

};

}

#endif //DYNSLAM_DYNSLAM_H
