
#ifndef DYNSLAM_DEPTHENGINE_H
#define DYNSLAM_DEPTHENGINE_H

#include <opencv/cv.h>

namespace dynslam {

/// \brief Contains the calibration parameters of a stereo rig, such as the AnnieWAY platform.
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
        using namespace std;
        cout << i << ", " << j << " " << disparity.cols << " x " << disparity.cols << endl;
        out_depth.at<short>(i, j) = DepthFromDisparity(disparity.at<short>(i, j), calibration);
      }
    }

  }

 protected:
  DepthEngine() {}
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHENGINE_H
