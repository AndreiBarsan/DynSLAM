

#include "InfiniTamDriver.h"

namespace dynslam { namespace drivers {

InfiniTamDriver::InfiniTamDriver(const ITMLibSettings *settings,
                                 const ITMRGBDCalib *calib,
                                 const Vector2i &imgSize_rgb,
                                 const Vector2i &imgSize_d)
  : ITMMainEngine(settings, calib, imgSize_rgb, imgSize_d) {}

void InfiniTamDriver::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage,
                                   ITMIMUMeasurement *imuMeasurement) {
  ITMMainEngine::ProcessFrame(rgbImage, rawDepthImage, imuMeasurement);
}


}}
