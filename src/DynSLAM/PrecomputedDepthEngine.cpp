
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
  if (hdr_depth_) {
    ReadFilePFM(out_disparity, depth_fpath);

    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Yes, not a lot of
    // programs can load them natively for manual inspection. Photoshop can,
//    cv::Mat preview(370, 1226, CV_8UC1);

    // FWIW it takes DispNet about 50 seconds to chew through a 1101 image kitti sequence at full
    // resolution (this includes the network initialization overhead AND dumping the images to the
    // disk). This means ~0.05s/frame, or 20 FPS.

    // Convert the 32-bit "HDR" depth into something we can display.
//    out_disparity.convertTo(preview, CV_8UC1, 1.0);
//
//    cv::imshow("PFM depth map preview", preview);
//    cv::waitKey(0);
  }
  else {
    out_disparity = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED);
  }

  this->frame_idx++;
}

float PrecomputedDepthEngine::DepthFromDisparity(const float disparity_px,
                                                 const StereoCalibration &calibration) {
  if (hdr_depth_) {
    return calibration.focal_length_px * calibration.baseline_meters / disparity_px;
  }
  else {
    // NOP, since we're reading low-res depthmaps directly for now
    return disparity_px;
  }
}

}
