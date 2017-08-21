
#ifndef DYNSLAM_DEPTHPROVIDER_H
#define DYNSLAM_DEPTHPROVIDER_H

#include <limits>

#include <opencv/cv.h>
#include "Utils.h"

// Some necessary forward declarations
namespace dynslam {
namespace utils {
std::string Type2Str(int type);
std::string Format(const std::string &fmt, ...);
}
}

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
  static constexpr float kMetersToMillimeters = 1000.0f;

 public:
  DepthProvider (const DepthProvider&) = default;
  DepthProvider (DepthProvider&&) = default;
  DepthProvider& operator=(const DepthProvider&) = default;
  DepthProvider& operator=(DepthProvider&&) = default;
  virtual ~DepthProvider() = default;

  /// \brief Computes a depth map from a stereo image pair (stereo -> disparity -> depth).
  virtual void DepthFromStereo(const cv::Mat &left,
                               const cv::Mat &right,
                               const StereoCalibration &calibration,
                               cv::Mat1s &out_depth,
                               float scale
  ) {
    if (input_is_depth_) {
      // Our input is designated as direct depth, not just disparity.
      DisparityMapFromStereo(left, right, out_depth);
      return;
    }

    // We need to compute disparity from stereo, and then depth from disparity.
    DisparityMapFromStereo(left, right, out_disparity_);

    // This should be templated in a nicer fashion...
    if (out_disparity_.type() == CV_32FC1) {
      DepthFromDisparityMap<float>(out_disparity_, calibration, out_depth, scale);
    } else if (out_disparity_.type() == CV_16SC1) {
      throw std::runtime_error("Cannot currently convert int16_t disparity to depth.");
    } else {
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

  // TODO-LOW(andrei): This can be sped up trivially using CUDA.
  /// \brief Computes a depth map from a disparity map using the `DepthFromDisparity` function at
  /// every pixel.
  /// \tparam T The type of the elements in the disparity input.
  /// \param disparity The disparity map.
  /// \param calibration The stereo calibration parameters used to compute depth from disparity.
  /// \param out_depth The output depth map, which gets populated by this method.
  /// \param scale Used to adjust the depth-from-disparity formula when using reduced-resolution
  ///              input. Unless evaluating the system's performance on low-res input, this should
  ///              be set to 1.
  template<typename T>
  void DepthFromDisparityMap(const cv::Mat_<T> &disparity,
                             const StereoCalibration &calibration,
                             cv::Mat1s &out_depth,
                             float scale
  ) {
    assert(disparity.size() == out_depth.size());
    assert(!input_is_depth_ && "Should not attempt to compute depth from disparity when the read "
        "data is already a depth map, and not just a disparity map.");

    // The max depth is an important factor for the quality of the resulting maps. Too big, and
    // our map will be very noisy; too small, and we only map the road and a couple of meters of
    // the sidewalks.
    int32_t min_depth_mm = static_cast<int32_t>(min_depth_m_ * kMetersToMillimeters);
    int32_t max_depth_mm = static_cast<int32_t>(max_depth_m_ * kMetersToMillimeters);

    // InfiniTAM requires short depth maps, so we need to ensure our depth can actually fit in a
    // short.
    int32_t max_representable_depth = std::numeric_limits<int16_t>::max();
    if (max_depth_mm >= max_representable_depth) {
      throw std::runtime_error(utils::Format("Unsupported maximum depth of %f meters (%d mm, "
                                                 "larger than the %d limit).", max_depth_m_,
                                             max_depth_mm, max_representable_depth));
    }

    for (int i = 0; i < disparity.rows; ++i) {
      for (int j = 0; j < disparity.cols; ++j) {
        T disp = disparity.template at<T>(i, j);
        int32_t depth_mm =
            static_cast<int32_t>(kMetersToMillimeters * scale * DepthFromDisparity(disp, calibration));

        if (abs(disp) < 1e-5) {
          depth_mm = 0;
        }

        if (depth_mm > max_depth_mm || depth_mm < min_depth_mm) {
          depth_mm = 0;
        }

        int16_t depth_mm_short = static_cast<int16_t>(depth_mm);
        out_depth.at<int16_t>(i, j) = depth_mm_short;
      }
    }
  }

  /// \brief The name of the technique being used for depth estimation.
  virtual const std::string &GetName() const = 0;

  float GetMinDepthMeters() const {
    return min_depth_m_;
  }

  void SetMinDepthMeters(float min_depth_m) {
    this->min_depth_m_ = min_depth_m;
  }

  float GetMaxDepthMeters() const {
    return max_depth_m_;
  }

  void SetMaxDepthMeters(float max_depth_m) {
    this->max_depth_m_ = max_depth_m;
  }

 protected:
  /// \param Whether the input is a depth map, or just a disparity map.
  /// \param min_depth_m The minimum depth, in meters, which is not considered too noisy.
  /// \param max_depth_m The maximum depth, in meters, which is not considered too noisy.
  explicit DepthProvider(bool input_is_depth, float min_depth_m, float max_depth_m) :
      input_is_depth_(input_is_depth),
      min_depth_m_(min_depth_m),
      max_depth_m_(max_depth_m) {}

  /// \brief If true, then assume the read maps are depth maps, instead of disparity maps.
  /// In this case, the depth from disparity computation is no longer performed.
  bool input_is_depth_;
  /// \brief Buffer in which the disparity map gets saved at every frame.
  cv::Mat out_disparity_;

 private:
  float min_depth_m_;
  float max_depth_m_;
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHPROVIDER_H
