
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthProvider.h"
#include "Utils.h"
#include "../pfmLib/ImageIOpfm.h"

namespace dynslam {

using namespace std;

const string kDispNetName = "precomputed-dispnet";
const string kPrecomputedElas = "precomputed-elas";

void PrecomputedDepthProvider::DisparityMapFromStereo(const cv::Mat &left,
                                                    const cv::Mat &right,
                                                    cv::Mat &out_disparity
) {
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.
  string depth_fpath = this->folder_ + "/" + utils::Format(this->fname_format_, this->frame_idx_);

  if (utils::EndsWith(depth_fpath, ".pfm")) {
    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Not a lot of
    // programs can load them natively for manual inspection. Photoshop can, but cannot natively
    // display the 32-bit image unless it is first set to 16-bit.
    ReadFilePFM(out_disparity, depth_fpath);
  } else {
    // Other formats we work with (png, pgm, etc.) can be read by OpenCV just fine.
    out_disparity = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED);
  }

  this->frame_idx_++;
}

float PrecomputedDepthProvider::DepthFromDisparity(const float disparity_px,
                                                 const StereoCalibration &calibration) {
  return calibration.focal_length_px * calibration.baseline_meters / disparity_px;
}

const string &PrecomputedDepthProvider::GetName() const {
  if (utils::EndsWith(fname_format_, "pfm")) {
    return kDispNetName;
  }
  else {
    return kPrecomputedElas;
  }
}

} // namespace dynslam
