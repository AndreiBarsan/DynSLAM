#ifndef DYNSLAM_DYNSLAM_H
#define DYNSLAM_DYNSLAM_H

#include <pangolin/display/opengl_render_state.h>
#include "ImageSourceEngine.h"

#include "InfiniTamDriver.h"
#include "InstRecLib/InstanceReconstructor.h"
#include "InstRecLib/PrecomputedSegmentationProvider.h"
#include "InstRecLib/SparseSFProvider.h"
#include "Input.h"

namespace dynslam {

using namespace instreclib;
using namespace instreclib::reconstruction;
using namespace instreclib::segmentation;
using namespace dynslam::drivers;

/// \brief The central class of the DynSLAM system.
/// It processes input stereo frames and generates separate maps for all encountered object
/// instances, as well one for the static background.
class DynSlam {

public:
  // TODO(andrei): If possible, get rid of the initialize method.
  void Initialize(InfiniTamDriver *itm_static_scene_engine,
                  SegmentationProvider *segmentation_provider,
                  SparseSFProvider *sparse_sf_provider);

  /// \brief Reads in and processes the next frame from the data source.
  void ProcessFrame(Input *input);

  const unsigned char* GetRaycastPreview() {
    // TODO(andrei): Get rid of reliance on itam enums via a driver abstraction.
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);
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

  const unsigned char* GetObjectRaycastPreview(int object_idx, const pangolin::OpenGlMatrix &model_view) {
    instance_reconstructor_->GetInstanceRaycastPreview(out_image_, object_idx, model_view);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  const unsigned char* GetObjectRaycastFreeViewPreview(
      int object_idx,
      const pangolin::OpenGlMatrix &model_view
   ) {
    // TODO(andrei): Finish implementing for actual objects. This now just works for static bg, but
    // we use 'GetObjectRaycastPreview' to get raycasts for instances, which is confusing name-wise.

    static_scene_->GetImage(
        out_image_,
        ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME,
        model_view);

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

  int GetInputWidth() {
    return input_width_;
  }

  int GetInputHeight() {
    return input_height_;
  }

  int GetCurrentFrameNo() {
    return current_frame_no_;
  }

  void SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) {
    string today_folder = utils::GetDate();
    system(utils::Format("mkdir -p mesh_out/%s/%s", dataset_name.c_str(), today_folder.c_str()).c_str());
    string map_fpath = utils::Format("mesh_out/%s/%s/static-%s-mesh.obj",
                                     dataset_name.c_str(),
                                     today_folder.c_str(),
                                     depth_name.c_str());
    cout << "Saving full static map to: " << map_fpath << endl;
    static_scene_->SaveSceneToMesh(map_fpath.c_str());
  }

  void SaveDynamicObject(const std::string &dataset_name, const std::string &depth_name, int object_id) {
    cout << "Saving mesh for object #" << object_id << "'s reconstruction..." << endl;
    string today_folder = utils::GetDate();

    // TODO(andrei): Make this more cross-platform and more secure.
    system(utils::Format("mkdir -p mesh_out/%s/%s", dataset_name.c_str(), today_folder.c_str()).c_str());

    string instance_fpath = utils::Format("mesh_out/%s/%s/instance-%s-%d-mesh.obj",
                                          dataset_name.c_str(),
                                          today_folder.c_str(),
                                          depth_name.c_str(),
                                          object_id);
    instance_reconstructor_->SaveObjectToMesh(object_id, instance_fpath);

    cout << "Done saving mesh for object #" << object_id << "'s reconstruction in file ["
         << instance_fpath << "]." << endl;
  }

  // Variants would solve this nicely, but they are C++17-only...
  // Will error out if no flow information is available.
  const SparseSceneFlow& GetLatestFlow() {
    return sparse_sf_provider_->GetFlow();
  }

private:
  // This is the main reconstruction component. Should split for dynamic+static.
  // In the future, we may need to write our own.
  // For now, this shall only handle reconstructing the static part of a scene.
  InfiniTamDriver *static_scene_;
  SegmentationProvider *segmentation_provider_;
  InstanceReconstructor *instance_reconstructor_;
  SparseSFProvider *sparse_sf_provider_;

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;

  int current_frame_no_;
  int input_width_;
  int input_height_;

  // TODO(andrei): Isolate this in a specific itam driver.
  const unsigned char* GetItamData(ITMMainEngine::GetImageType image_type) {
    static_scene_->GetImage(out_image_, image_type);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }
};

}

#endif //DYNSLAM_DYNSLAM_H
