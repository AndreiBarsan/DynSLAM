
#ifndef DYNSLAM_DEPTHPROVIDER_H
#define DYNSLAM_DEPTHPROVIDER_H

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

/// \brief ABC for components computing depth from stereo image pairs.
/// \note The methods of this interface are designed according to the OpenCV API style, and
/// return their results into pre-allocated out parameters.
class DepthProvider {
 public:
  virtual ~DepthProvider() {}

  /// \brief Computes a depth map from a stereo image pair (stereo -> disparity -> depth).
  void DepthFromStereo(const cv::Mat &left,
                       const cv::Mat &right,
                       const StereoCalibration &calibration,
                       cv::Mat1s &out_depth) {
    if(input_is_depth_) {
      // Our input is designated as direct depth, not just disparity.
      DisparityMapFromStereo(left, right, out_depth);
      return;
    }

    // We need to compute disparity from stereo, and then depth from disparity.
    DisparityMapFromStereo(left, right, out_disparity_);

    // This should be templated in a nicer fashion...
    if (out_disparity_.type() == CV_32FC1) {
      DepthFromDisparityMap<float>(out_disparity_, calibration, out_depth);
    }
    else if (out_disparity_.type() == CV_16SC1) {
      DepthFromDisparityMap<uint16_t>(out_disparity_, calibration, out_depth);
    }
    else {
      throw std::runtime_error(utils::Format(
          "Unknown data type for disparity matrix [%s]. Supported are CV_32FC1 and CV_16SC1.",
          utils::Type2Str(out_disparity_.type()).c_str()
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

  // TODO(andrei): Higher max depth once we're modeling noise properly.
  // TODO-LOW(andrei): This can be sped up trivially using CUDA.
  /// \brief Computes a depth map from a disparity map using the `DepthFromDisparity` function at
  /// every pixel.
  /// \tparam T The type of the elements in the disparity input.
  /// \param disparity The disparity map.
  /// \param calibration The stereo calibration parameters used to compute depth from disparity.
  /// \param out_depth The output depth map, which gets populated by this method.
  /// \param min_depth_m The minimum depth, in meters, which is not considered too noisy.
  /// \param max_depth_m The maximum depth, in meters, which is not considered too noisy.
  template<typename T>
  void DepthFromDisparityMap(const cv::Mat_<T> &disparity,
                             const StereoCalibration &calibration,
                             cv::Mat1s &out_depth,
                             float min_depth_m =  0.50f,
                             float max_depth_m = 20.0f
  ) {
    assert(disparity.size() == out_depth.size());

    for(int i = 0; i < disparity.rows; ++i) {
      for(int j = 0; j < disparity.cols; ++j) {
        T disp = disparity.template at<T>(i, j);

        float kMetersToMillimeters = 1000.0;
        int32_t depth_mm = static_cast<int32_t>(kMetersToMillimeters * DepthFromDisparity(disp, calibration));

        // The max depth is an important factor for the quality of the resulting maps. Too big, and
        // our map will be very noisy; too small, and we only map the road and a couple of meters of
        // the sidewalks.
        int32_t min_depth_mm = static_cast<int32_t>(min_depth_m * kMetersToMillimeters);

        if (depth_mm > max_depth_m * kMetersToMillimeters || depth_mm < min_depth_mm) {
          depth_mm = std::numeric_limits<int16_t>::max();
        }

        int16_t depth_mm_short = static_cast<int16_t>(depth_mm);
        out_depth.at<int16_t>(i, j) = depth_mm_short;
      }
    }
  }

  /// \brief The name of the technique being used for depth estimation.
  virtual const std::string& GetName() const = 0;

 protected:
  DepthProvider(bool input_is_depth) : input_is_depth_(input_is_depth) {}

  /// \brief If true, then assume the read maps are depth maps, instead of disparity maps.
  /// In this case, the depth from disparity computation is no longer performed.
  bool input_is_depth_;
  /// \brief Buffer in which the disparity map gets saved at every frame.
  cv::Mat out_disparity_;
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHPROVIDER_H
