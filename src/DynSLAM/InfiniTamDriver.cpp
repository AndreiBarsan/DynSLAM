

#include "InfiniTamDriver.h"

namespace dynslam {
namespace drivers {

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix) {
  Matrix4f M;
  for(int i = 0; i < 16; ++i) {
    M.m[i] = static_cast<float>(pangolin_matrix.m[i]);
  }

  ITMPose itm_pose;
  itm_pose.SetM(M);
  itm_pose.Coerce();

  return itm_pose;
}

InfiniTamDriver::InfiniTamDriver(//const string& dataset_root,
                                 const ITMLibSettings* settings,
                                 const ITMRGBDCalib* calib,
                                 const Vector2i& imgSize_rgb,
                                 const Vector2i& imgSize_d)
  : ITMMainEngine(settings, calib, imgSize_rgb, imgSize_d) {}

// TODO(andrei): Get rid of these notes.
// We will likely want different alignment techniques for the background and for the dynamic
// objects. Without the 'GetImage' stuff, the ITMMainEngine code is ~150 LOC. Their architecture is
// very clean; the main engine only connects other core components. We could therefore simply just
// use their denseMapper directly, and solve the other bits ourselves: ground truth or sparse+dense
// VO for the background, SF-based+dense VO for the individual objects, etc.


void InfiniTamDriver::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage,
                                   ITMIMUMeasurement *imuMeasurement) {
  if (imuMeasurement != nullptr) {
    throw runtime_error("IMU data integration not supported.");
  }

  this->UpdateView(rgbImage, rawDepthImage);
  this->Track();
  this->Integrate();
  this->PrepareNextStep();
}

} // namespace drivers}
} // namespace dynslam
