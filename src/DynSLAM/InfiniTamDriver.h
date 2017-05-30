

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

#include <iostream>

#include <pangolin/pangolin.h>
#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"
#include "ImageSourceEngine.h"

namespace dynslam {
namespace drivers {

/// \brief Interfaces between DynSLAM and InfiniTAM.
class InfiniTamDriver : public ITMMainEngine {
public:
  // TODO(andrei): We may need to add another layer of abstraction above the drivers to get the best
  // modularity possible.
  static InfiniTamDriver* Build(const string &dataset_root, ImageSourceEngine** image_source) {
    ITMLibSettings *settings = new ITMLibSettings();

    const string calib_fpath = dataset_root + "/itm-calib.txt";
    const string rgb_image_format = dataset_root + "/precomputed-depth/Frames/%04i.ppm";
    const string depth_image_format = dataset_root + "/precomputed-depth/Frames/%04i.pgm";

    *image_source = new ImageFileReader(
        calib_fpath.c_str(),
        rgb_image_format.c_str(),
        depth_image_format.c_str()
    );

    InfiniTamDriver *driver = new InfiniTamDriver(settings,
                                                  new ITMRGBDCalib((*image_source)->calib),
                                                  (*image_source)->getRGBImageSize(),
                                                  (*image_source)->getDepthImageSize());

    return driver;
  }

  InfiniTamDriver(const ITMLibSettings* settings,
                  const ITMRGBDCalib* calib,
                  const Vector2i& imgSize_rgb,
                  const Vector2i& imgSize_d);

  void ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage,
                    ITMIMUMeasurement *imuMeasurement = nullptr) override;

  void UpdateView(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage) {
    // * If 'view' is null, this allocates its RGB and depth buffers.
    // * Afterwards, it converts the depth map we give it into a float depth map (we may be able to
    //   skip this step in our case, since we have control over how our depth map is computed).
    // * It then filters the shit out of the depth map (maybe we could skip this?) using five steps
    //   of bilateral filtering.
    this->viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter,
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
  }

  const ITMLibSettings* GetSettings() const {
    return settings;
  }

  // Not const because 'ITMMainEngine' is not const either.
  void GetImage(
      ITMUChar4Image *out,
      GetImageType get_image_type,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix()
  ) {
    // TODO helper function for this
    Matrix4f M;
    for(int i = 0; i < 16; ++i) {
      M.m[i] = static_cast<float>(model_view.m[i]);
    }

    ITMPose itm_freeview_pose;
    itm_freeview_pose.SetM(M);

//    using namespace std;
//    std::cout << endl << "M:" << endl << itm_freeview_pose.GetM() << std::endl;
//    itm_freeview_pose.Coerce();
//    std::cout << itm_freeview_pose.GetM().

    if (nullptr != this->view) {
      if(nullptr == this->view->calib) {
        // TODO(andrei): Check this out.. it seems to happen when a car's trail gets collected, but
        // its reconstruction stays in memory. I think the view gets deallocated and this->view
        // becomes stale.
        cout << "Unexpected for view to be OK but calib nil. " << endl;
        return;
      }

      ITMIntrinsics intrinsics = this->view->calib->intrinsics_d;
      ITMMainEngine::GetImage(
          out,
          get_image_type,
          &itm_freeview_pose,
          &intrinsics);
    }
    else {
      std::cerr << "Warning: no raycast available yet." << endl;
    }
  }

};

} // namespace drivers
} // namespace dynslam


#endif //DYNSLAM_INFINITAMDRIVER_H
