
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
    // Otherwise load an OpenCV XML dump (since we need 16-bit signed depth maps, which OpenCV
    // cannot save as regular images, even though the PNG spec has nothing against them).
    cv::FileStorage fs(depth_fpath, cv::FileStorage::READ);
    if(!fs.isOpened()) {
      throw std::runtime_error("Could not read precomputed depth map.");
    }
    fs["depth-frame"] >> out;
    if (out.type() != CV_16SC1) {
      throw std::runtime_error("Precomputed depth map had the wrong format.");
    }
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
    int16_t max_depth_mm_s = static_cast<int16_t>(round(max_depth_mm_f));

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
          int16_t depth = out.at<int16_t>(i, j);
          if (depth > max_depth_mm_s) {
            out.at<int16_t>(i, j) = 0;
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
