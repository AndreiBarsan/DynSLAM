
#ifndef DYNSLAM_DEPTHENGINE_H
#define DYNSLAM_DEPTHENGINE_H

#include <opencv/cv.h>
#include "Utils.h"

namespace dynslam { namespace utils { // voodoo
std::string type2str(int type);
}}

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
    assert(disparity.size() == out_depth.size());
    using namespace std;

    cout << disparity.size() << out_depth.size() << endl;
    cout << disparity.elemSize() << " " << disparity.type() << " "
         << dynslam::utils::type2str(disparity.type()) << endl;
    for(int i = 0; i < disparity.rows; ++i) {
      for(int j = 0; j < disparity.cols; ++j) {
//        cout << i << ", " << j << " " << disparity.cols << " x " << disparity.cols << endl;
        // TODO correctly use the right way depending on whether the input image is short or float
        float disp = disparity.at<float>(i, j);

        int32_t disp_long = static_cast<int32_t>(1000.0 * DepthFromDisparity(disp, calibration));
        if (disp_long > 20 * 1000 || disp_long < 1000) {
          disp_long = numeric_limits<uint16_t>::max();
        }
////
        uint16_t depth_s = static_cast<uint16_t>(disp_long);
        // Use for oldschoold pgm depth files
//        uint16_t depth_s = disparity.at<uint16_t>(i, j);
        out_depth.at<uint16_t>(i, j) = depth_s;
      }
    }
    cout << "Loop done." << endl;

  }

 protected:
  DepthEngine() {}
};

} // namespace dynslam

#endif //DYNSLAM_DEPTHENGINE_H
