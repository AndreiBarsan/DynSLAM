
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthEngine.h"
#include "Utils.h"

namespace dynslam {

using namespace std;

void PrecomputedDepthEngine::DisparityMapFromStereo(const cv::Mat &left,
                                                    const cv::Mat &right,
                                                    cv::Mat &out_disparity
) {
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.
  string depth_fpath = this->folder + "/" + utils::Format(this->fname_format, this->frame_idx);
  out_disparity = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED);

  this->frame_idx++;
}

float PrecomputedDepthEngine::DepthFromDisparity(const float disparity_px,
                                                 const StereoCalibration &calibration) {
  // NOP, since we're reading depthmaps directly for now
  return disparity_px;
}

}
