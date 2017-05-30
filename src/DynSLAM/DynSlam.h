#ifndef DYNSLAM_DYNSLAM_H
#define DYNSLAM_DYNSLAM_H

#include <pangolin/display/opengl_render_state.h>
#include "ImageSourceEngine.h"

#include "InfiniTamDriver.h"
#include "InstRecLib/InstanceReconstructor.h"
#include "InstRecLib/PrecomputedSegmentationProvider.h"

namespace dynslam {

using namespace instreclib::reconstruction;
using namespace instreclib::segmentation;
using namespace dynslam::drivers;

/// \brief The central class of the DynSLAM system.
/// It processes input stereo frames and generates separate maps for all encountered object
/// instances, as well one for the static background.
class DynSlam {

public:
  void Initialize(InfiniTamDriver* itm_static_scene_engine_,
                    ImageSourceEngine* image_source,
                    SegmentationProvider* segmentation_provider);

  /// \brief Reads in and processes the next frame from the data source.
  void ProcessFrame();

  const unsigned char* GetRaycastPreview() {
    // TODO(andrei): Get rid of reliance on itam enums via a driver abstraction.
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);
  }

  /// \brief Returns an **RGBA** preview of the latest color frame.
  const unsigned char* GetRgbPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_RGB);
  }

  /// \brief Returns an **RGBA** preview of the latest depth frame.
  const unsigned char* GetDepthPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_DEPTH);
  }

  const unsigned char* GetObjectRaycastPreview(int object_idx, const pangolin::OpenGlMatrix &model_view) {
    instance_reconstructor_->GetInstanceRaycastPreview(out_image_, object_idx, model_view);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  const unsigned char* GetObjectRaycastFreeViewPreview(
      int object_idx,
      const pangolin::OpenGlMatrix &model_view
   ) {

    // TODO(andrei): Finish implementing for actual objects. This now just works for static bg.

    static_scene_->GetImage(
        out_image_,
        ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME,
        model_view);

    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  /// \brief Returns an RGBA unsigned char frame containing the preview of the most recent frame's
  /// semantic segmentation.
  const unsigned char* GetSegmentationPreview() {
    return segmentation_provider_->GetSegResult()->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  /// \brief Returns an **RGBA** preview of the latest segmented object instance.
  const unsigned char* GetObjectPreview(int object_idx);

  InstanceReconstructor* GetInstanceReconstructor() {
    return instance_reconstructor_;
  }

  int GetInputWidth() {
    return image_source_->getDepthImageSize().width;
  }

  int GetInputHeight() {
    return image_source_->getDepthImageSize().height;
  }

  int GetCurrentFrameNo() {
    return current_frame_no_;
  }

  void SaveStaticMap() {
    // TODO(andrei): Custom name, etc.
    static_scene_->SaveSceneToMesh("mesh_out.stl");
  }

private:
  ITMLibSettings itm_lib_settings_;
  // TODO(andrei): Write custom image source.
  ImageSourceEngine *image_source_;

  // This is the main reconstruction component. Should split for dynamic+static.
  // In the future, we may need to write our own.
  // For now, this shall only handle reconstructing the static part of a scene.
  InfiniTamDriver *static_scene_;

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  ITMUChar4Image *input_rgb_image_;
  ITMShortImage  *input_raw_depth_image_;

  int current_frame_no_;

  Vector2i window_size_;

  // TODO(andrei): Isolate this in a specific itam driver.
  const unsigned char* GetItamData(ITMMainEngine::GetImageType image_type) {
    static_scene_->GetImage(out_image_, image_type);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  SegmentationProvider *segmentation_provider_;
  InstanceReconstructor *instance_reconstructor_;
};

}

#endif //DYNSLAM_DYNSLAM_H
