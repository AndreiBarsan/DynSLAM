
#ifndef DYNSLAM_DEPTHENGINE_H
#define DYNSLAM_DEPTHENGINE_H

#include <opencv/cv.h>

namespace dynslam {

/// \brief Contains the calibration parameters of a stereo rig, such as the AnnieWAY platform.
struct StereoCalibration {
  float baseline_meters;
  float focal_length_px;
};

/// \brief Interface for components computing depth from stereo image pairs.
/// \note The methods of this interface are designed according to the OpenCV API style, and
/// return their results into pre-allocated out parameters.
class DepthEngine {
 public:
  /// \brief Computes a disparity map from
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
  virtual void DepthFromDisparityMap(const cv::Mat &disparity,
                                     const StereoCalibration &calibration,
                                     cv::Mat &out_depth) {
    for(int i = 0; i < disparity.rows; ++i) {
      for(int j = 0; j < disparity.cols; ++j) {
        out_depth.at(i, j) = DepthFromDisparity(disparity.at(i, j), calibration);
      }
    }

  }
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHENGINE_H
