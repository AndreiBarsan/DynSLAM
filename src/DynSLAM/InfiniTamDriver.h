

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

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

  const ITMLibSettings* GetSettings() const {
    return settings;
  }

//  const string& GetDatasetRoot() const {
//    return dataset_root_;
//  }

 private:
//  string dataset_root_;
};

} // namespace drivers
} // namespace dynslam


#endif //DYNSLAM_INFINITAMDRIVER_H
