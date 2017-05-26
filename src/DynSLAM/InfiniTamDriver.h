

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"

namespace dynslam { namespace drivers {

// TODO(andrei): Add common driver interface behind which to possibly

class InfiniTamDriver : public ITMMainEngine {
public:
  InfiniTamDriver(const ITMLibSettings *settings, const ITMRGBDCalib *calib,
                  const Vector2i &imgSize_rgb, const Vector2i &imgSize_d);

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

  void Track() {
    this->trackingController->Track(this->trackingState, this->view);
  }

  void Integrate() {
    this->denseMapper->ProcessFrame(
      // For separate integrations we'd need to compute the tracking state appropriately.
      // We'd also prolly need a custom scene, and a custom renderState_live for each object.
      // The scene actually holds the voxel hash. It's almost a POD.
      // The render state is used for things like raycasting

      // We already generate our new view when splitting the input based on the segmentation.
      this->view, this->trackingState, this->scene, this->renderState_live);
  }

  void PrepareNextStep() {
    // This may not be necessary if we're using ground truth VO.
    this->trackingController->Prepare(this->trackingState, this->view, this->renderState_live);
  }

};

}}


#endif //DYNSLAM_INFINITAMDRIVER_H
