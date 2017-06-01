
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

  string frame_depth_fpath = this->folder + "/" +
      utils::Format(this->fname_format, this->frame_idx);
  cout << "Reading depth from file " << frame_depth_fpath << endl;
  cv::imread(frame_depth_fpath);

  this->frame_idx++;
}

float PrecomputedDepthEngine::DepthFromDisparity(const float disparity_px,
                                                 const StereoCalibration &calibration) {
  // NOP, since we're reading depthmaps directly for now
  return disparity_px;
}

}
