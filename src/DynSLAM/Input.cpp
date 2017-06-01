
#include "Input.h"

namespace dynslam {

ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const std::string &fpath) {
  ITMLib::Objects::ITMRGBDCalib out_calib;
  if (! ITMLib::Objects::readRGBDCalib(fpath.c_str(), out_calib)) {
    throw std::runtime_error(dynslam::utils::Format(
        "Could not read calibration file: [%s]\n", fpath));
  }
  return out_calib;
}
}
