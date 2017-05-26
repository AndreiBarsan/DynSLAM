

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"

namespace dynslam { namespace drivers {

//using namespace ITMLib::Engine;

class InfiniTamDriver : public ITMMainEngine {
public:
  InfiniTamDriver(const ITMLibSettings *settings, const ITMRGBDCalib *calib,
                  const Vector2i &imgSize_rgb, const Vector2i &imgSize_d);

  void ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage,
                    ITMIMUMeasurement *imuMeasurement = nullptr) override;

};

}}


#endif //DYNSLAM_INFINITAMDRIVER_H
