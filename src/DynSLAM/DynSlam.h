#ifndef DYNSLAM_DYNSLAM_H
#define DYNSLAM_DYNSLAM_H


#include "ImageSourceEngine.h"

#include "../InfiniTAM/InfiniTAM/InstRecLib/InstanceReconstructor.h"

namespace dynslam {

using namespace InstRecLib::Reconstruction;

/// \brief The central class of the DynSLAM system.
/// It processes input stereo frames and generates separate maps for all encountered object
/// instances, as well one for the static background.
class DynSlam {

public:
  void Initialize(ITMMainEngine *itm_static_scene_engine_, ImageSourceEngine *image_source);

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

  /// \brief Returns an RGBA unsigned char frame containing the preview of the most recent frame's
  /// semantic segmentation.
  const unsigned char* GetSegmentationPreview() {
    return GetItamData(ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SEGMENTATION_RESULT);
  }

  /// \brief Returns an **RGBA** preview of the latest segmented object instance.
  const unsigned char* GetObjectPreview(int object_idx);

  InstanceReconstructor* GetInstanceReconstructor() {
    return itm_static_scene_engine_->GetInstanceReconstructor();
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

private:
  ITMLibSettings itm_lib_settings_;
  // TODO(andrei): Write custom image source.
  ImageSourceEngine *image_source_;

  // This is the main reconstruction component. Should split for dynamic+static.
  // In the future, we may need to write our own.
  // For now, this shall only handle reconstructing the static part of a scene.
  ITMMainEngine *itm_static_scene_engine_;

  ITMUChar4Image *out_image_;
  ITMUChar4Image *input_rgb_image_;
  ITMShortImage  *input_raw_depth_image_;

  int current_frame_no_;

  Vector2i window_size_;

  // TODO(andrei): Isolate this in a specific itam driver.
  const unsigned char* GetItamData(ITMMainEngine::GetImageType image_type) {
    itm_static_scene_engine_->GetImage(out_image_, image_type);
    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }
};

}

#endif //DYNSLAM_DYNSLAM_H
