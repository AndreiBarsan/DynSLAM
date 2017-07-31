
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthProvider.h"
#include "Utils.h"
#include "../pfmLib/ImageIOpfm.h"

namespace dynslam {

using namespace std;

const string kDispNetName = "precomputed-dispnet";
const string kPrecomputedElas = "precomputed-elas";

void PrecomputedDepthProvider::DisparityMapFromStereo(const cv::Mat&,
                                                      const cv::Mat&,
                                                      cv::Mat &out_disparity
) {
  ReadPrecomputed(this->input_->GetCurrentFrame(), out_disparity);
}

void PrecomputedDepthProvider::ReadPrecomputed(int frame_idx, cv::Mat &out) const {
  // TODO(andrei): Read correct precomputed depth when doing evaluation of arbitrary frames.
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.
  string depth_fpath = this->folder_ + "/" + utils::Format(this->fname_format_, frame_idx);

  if (utils::EndsWith(depth_fpath, ".pfm")) {
    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Not a lot of programs
    // can load them natively for manual inspection. Photoshop can, but cannot natively display the
    // 32-bit image unless it is first converted to 16-bit.
    ReadFilePFM(out, depth_fpath);
  } else {
    // Other formats we work with (png, pgm, etc.) can be read by OpenCV just fine.
    out = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED);
  }

  if (out.cols == 0 || out.rows == 0) {
    throw runtime_error(utils::Format(
        "Could not read precomputed depth map from file [%s]. Please check that the file exists, "
            "and is a readable, valid, image.",
        depth_fpath.c_str()));
  }

  if (this->input_is_depth_) {
    // We're reading depth directly, so we need to ensure the max depth here.
    float max_depth_mm_f = GetMaxDepthMeters() * kMetersToMillimeters;
    uint16_t max_depth_mm_ui = static_cast<uint16_t>(round(max_depth_mm_f));

    for(int i = 0; i < out.rows; ++i) {
      for(int j = 0; j < out.cols; ++j) {
        // TODO-LOW(andrei): Do this in a nicer way...
        if(out.type() == CV_32FC1) {
          float depth = out.at<float>(i, j);
          if (depth > max_depth_mm_f) {
            out.at<float>(i, j) = 0.0f;
          }
        }
        else {
          uint16_t depth = out.at<uint16_t>(i, j);
          if (depth > max_depth_mm_ui) {
            out.at<uint16_t>(i, j) = 0;
          }
        }
      }
    }
  }

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
