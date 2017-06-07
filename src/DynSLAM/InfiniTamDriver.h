

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

#include <opencv/cv.h>
#include <iostream>

#include <pangolin/pangolin.h>
#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"
#include "ImageSourceEngine.h"
#include "Input.h"

namespace dynslam {
namespace drivers {

// Utilities for converting from OpenCV to InfiniTAM vectors.
template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Vec<T, 2> in) {
  return ORUtils::Vector2<T>(in[0], in[1]);
}

template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Size_<T> in) {
  return ORUtils::Vector2<T>(in.width, in.height);
}

template<typename T>
ORUtils::Vector3<T> ToItmVec(const cv::Vec<T, 3> in) {
  return ORUtils::Vector3<T>(in[0], in[1], in[2]);
}

template<typename T>
ORUtils::Vector4<T> ToItmVec(const cv::Vec<T, 4> in) {
  return ORUtils::Vector4<T>(in[0], in[1], in[2], in[3]);
}

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix, bool flip_ud = true, bool fix_roll = true);

/// \brief Interfaces between DynSLAM and InfiniTAM.
class InfiniTamDriver : public ITMMainEngine {
public:
  // TODO(andrei): We may need to add another layer of abstraction above the drivers to get the best
  // modularity possible (driver+inpu+custom settings combos).

  InfiniTamDriver(
      const ITMLibSettings *settings,
      const ITMRGBDCalib *calib,
      const Vector2i &img_size_rgb,
      const Vector2i &img_size_d)
      : ITMMainEngine(settings, calib, img_size_rgb, img_size_d),
        rgb_itm_(new ITMUChar4Image(img_size_rgb, true, true)),
        raw_depth_itm_(new ITMShortImage(img_size_d, true, true)),
        rgb_cv_(new cv::Mat3b(img_size_rgb.height, img_size_rgb.width)),
        raw_depth_cv_(new cv::Mat1s(img_size_d.height, img_size_d.width))
  { }

  virtual ~InfiniTamDriver() {
    delete rgb_itm_;
    delete raw_depth_itm_;
    delete rgb_cv_;
    delete raw_depth_cv_;
  }

  // TODO(andrei): I was passing a Mat4b but didnt even get a warning. Is that normal? I was
  // expecting a little more type safety...
  void UpdateView(const cv::Mat3b &rgb_image, const cv::Mat_<uint16_t> &raw_depth_image) {
    CvToItm(rgb_image, rgb_itm_);
    CvToItm(raw_depth_image, raw_depth_itm_);

    // * If 'view' is null, this allocates its RGB and depth buffers.
    // * Afterwards, it converts the depth map we give it into a float depth map (we may be able to
    //   skip this step in our case, since we have control over how our depth map is computed).
    // * It then filters the shit out of the depth map (maybe we could skip this?) using five steps
    //   of bilateral filtering.
    this->viewBuilder->UpdateView(&view, rgb_itm_, raw_depth_itm_, settings->useBilateralFilter,
                                  settings->modelSensorNoise);
  }

  // used by the instance reconstruction
  void SetView(ITMView *view) {
    if (this->view) {
      // TODO(andrei): These views should be memory managed by the tracker. Make sure this is right.
//      delete this->view;
    }

    this->view = view;
  }

  void Track() {
    // Consider leveraging sparse scene flow here, for dynamic instances, maybe?
    this->trackingController->Track(this->trackingState, this->view);
  }

  void Integrate() {
    this->denseMapper->ProcessFrame(
      // We already generate our new view when splitting the input based on the segmentation.
      // The tracking state is kept up-to-date by the tracker.
      // The scene actually holds the voxel hash. It's almost a vanilla struct.
      // The render state is used for things like raycasting
      this->view, this->trackingState, this->scene, this->renderState_live);
  }

  void PrepareNextStep() {
    // This may not be necessary if we're using ground truth VO.
    this->trackingController->Prepare(this->trackingState, this->view, this->renderState_live);

    // Keep the OpenCV previews up to date.
    ItmToCv(*this->view->rgb, rgb_cv_);
    ItmToCv(*this->view->depth, raw_depth_cv_);
  }

  const ITMLibSettings* GetSettings() const {
    return settings;
  }

  // Not const because 'ITMMainEngine's implementation is not const either.
  void GetImage(
      ITMUChar4Image *out,
      GetImageType get_image_type,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix()
  );

  /// \brief Returns the RGB "seen" by this particular InfiniTAM instance.
  /// This may not be the full original RGB frame due to, e.g., masking.
  const cv::Mat3b* GetRgbPreview() const {
    return rgb_cv_;
  }

  /// \brief Returns the depth "seen" by this particular InfiniTAM instance.
  /// This may not be the full original depth frame due to, e.g., masking.
  const cv::Mat1s* GetDepthPreview() const {
    return raw_depth_cv_;
  }

 private:
  ITMUChar4Image *rgb_itm_;
  ITMShortImage  *raw_depth_itm_;

  cv::Mat3b *rgb_cv_;
  cv::Mat1s *raw_depth_cv_;
};

} // namespace drivers
} // namespace dynslam


#endif //DYNSLAM_INFINITAMDRIVER_H
