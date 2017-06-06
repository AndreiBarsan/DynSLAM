
#ifndef DYNSLAM_DEPTHENGINE_H
#define DYNSLAM_DEPTHENGINE_H

#include <limits>

#include <opencv/cv.h>
#include "Utils.h"

// Some necessary forward declarations
namespace dynslam { namespace utils {
  std::string Type2Str(int type);
  std::string Format(const std::string& fmt, ...);
}}

namespace dynslam {

/// \brief Contains the calibration parameters of a stereo rig, such as the AnnieWAY platform,
/// which is the one used to record the KITTI dataset.
struct StereoCalibration {
  float baseline_meters;
  float focal_length_px;

  StereoCalibration(float baseline_meters, float focal_length_px)
      : baseline_meters(baseline_meters), focal_length_px(focal_length_px) {}
};

/// \brief Interface for components computing depth from stereo image pairs.
/// \note The methods of this interface are designed according to the OpenCV API style, and
/// return their results into pre-allocated out parameters.
class DepthEngine {
 public:
  virtual ~DepthEngine() {}

  /// \brief Computes a depth map from a stereo image pair (stereo -> disparity -> depth).
  void DepthFromStereo(const cv::Mat &left,
                       const cv::Mat &right,
                       const StereoCalibration &calibration,
                       cv::Mat_<uint16_t> &out_depth) {
    if(input_is_depth_) {
      DisparityMapFromStereo(left, right, out_depth);
      return;
    }

    // TODO(andrei): Consider reusing this buffer.
    cv::Mat out_disparity;
    DisparityMapFromStereo(left, right, out_disparity);

    // ...and this one?
    out_depth = cv::Mat_<uint16_t>(out_disparity.size(), CV_16UC1);

    // This should be templated in a nicer fashion...
    if (out_disparity.type() == CV_32FC1) {
      DepthFromDisparityMap<float>(out_disparity, calibration, out_depth);
    }
    else if (out_disparity.type() == CV_16UC1) {
      DepthFromDisparityMap<uint16_t>(out_disparity, calibration, out_depth);
    }
    else {
      throw std::runtime_error(utils::Format(
          "Unknown data type for disparity matrix [%s]. Supported are CV_32FC1 and CV_16UC1.",
          utils::Type2Str(out_disparity.type()).c_str()
      ));
    }
  }

  /// \brief Computes a disparity map from a stereo image pair.
  virtual void DisparityMapFromStereo(const cv::Mat &left,
                                      const cv::Mat &right,
                                      cv::Mat &out_disparity) = 0;

  /// \brief Converts a single disparity pixel value to a depth value expressed in meters.
  virtual float DepthFromDisparity(const float disparity_px,
                                   const StereoCalibration &calibration) {
    return (calibration.baseline_meters * calibration.focal_length_px) / disparity_px;
  }

  // TODO(andrei): This can be sped up trivially using CUDA.
  /// \brief Computes a depth map from a disparity map using the `DepthFromDisparity` function at
  /// every pixel.
  template<typename T>
  void DepthFromDisparityMap(const cv::Mat_<T> &disparity,
                             const StereoCalibration &calibration,
                             cv::Mat_<uint16_t> &out_depth) {
    assert(disparity.size() == out_depth.size());

    for(int i = 0; i < disparity.rows; ++i) {
      for(int j = 0; j < disparity.cols; ++j) {
        // BURN THE WITCH
        T disp = disparity.template at<T>(i, j);

        double kMetersToMillimeters = 1000.0;
        int32_t depth_mm = static_cast<int32_t>(kMetersToMillimeters * DepthFromDisparity(disp, calibration));

        // TODO(andrei): Pass this as a parameter.
        // This is an important factor for the quality of the resulting maps. Too big, and our map
        // will be very noisy; too small, and we only map the road and a couple of meters of the
        // sidewalks.
        const int32_t kMaxDepthMeters = 12;

        // TODO(andrei): Log min/max/mean depth and other stats, and verify whether the disparities
        // produced by dispnet are consistent across frames.

        if (depth_mm > kMaxDepthMeters * kMetersToMillimeters || depth_mm < 0) {
          depth_mm = std::numeric_limits<uint16_t>::max();
        }

        uint16_t depth_mm_short = static_cast<uint16_t>(depth_mm);
        out_depth.at<uint16_t>(i, j) = depth_mm_short;
      }
    }
  }

 protected:
  DepthEngine(bool input_is_depth) : input_is_depth_(input_is_depth) {}

  /// \brief If true, then assume the read maps are depth maps, instead of disparity maps.
  /// In this case, the depth from disparity computation is no longer performed.
  bool input_is_depth_;
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHENGINE_H
