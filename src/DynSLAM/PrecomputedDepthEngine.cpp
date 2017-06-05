
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthEngine.h"
#include "Utils.h"
#include "../pfmLib/ImageIOpfm.h"

namespace dynslam {

using namespace std;

void PrecomputedDepthEngine::DisparityMapFromStereo(const cv::Mat &left,
                                                    const cv::Mat &right,
                                                    cv::Mat &out_disparity
) {
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.
  string depth_fpath = this->folder + "/" + utils::Format(this->fname_format, this->frame_idx);

  if (utils::EndsWith(depth_fpath, ".pfm")) {
    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Not a lot of
    // programs can load them natively for manual inspection. Photoshop can, but cannot natively
    // display the 32-bit image unless it is first set to 16-bit.
    ReadFilePFM(out_disparity, depth_fpath);
  } else {
    // Other formats we work with (png, pgm, etc.) can be read by OpenCV just fine.
    out_disparity = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED);
  }

  this->frame_idx++;
}

float PrecomputedDepthEngine::DepthFromDisparity(const float disparity_px,
                                                 const StereoCalibration &calibration) {
  return calibration.focal_length_px * calibration.baseline_meters / disparity_px;
}

} // namespace dynslam
