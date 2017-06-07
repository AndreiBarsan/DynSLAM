
#ifndef DYNSLAM_INPUT_H
#define DYNSLAM_INPUT_H

#include <string>
#include <highgui.h>

#include "DepthProvider.h"
#include "Utils.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMRGBDCalib.h"

namespace dynslam {

/// \brief Provides input from DynSLAM, in the form of RGBD frames.
/// Since DynSLAM currently operates with stereo input, this class also computes depth from stereo.
/// Currently, this is "computed" by reading the depth maps from disk, but the plan is to compute
/// depth on the fly in the future.
class Input {
 public:
  Input(const std::string &dataset_folder,
        DepthProvider *depth_provider,
        const ITMLib::Objects::ITMRGBDCalib &calibration,
        const StereoCalibration &stereo_calibration)
      : depth_provider_(depth_provider),
        dataset_folder_(dataset_folder),
        frame_idx_(0),
        calibration_(calibration),
        stereo_calibration_(stereo_calibration) {}

  bool HasMoreImages();

  /// \brief Advances the input reader to the next frame.
  /// \returns True if the next frame's files could be read successfully.
  bool ReadNextFrame();

  /// \brief Returns pointers to the latest RGB and depth data.
  void GetCvImages(cv::Mat3b **rgb, cv::Mat1s **raw_depth);

  /// \brief Returns pointers to the latest grayscale input frames.
  void GetCvStereoGray(cv::Mat1b **left, cv::Mat1b **right);

  cv::Size2i GetRgbSize() const {
    return cv::Size2i(static_cast<int>(calibration_.intrinsics_rgb.sizeX),
                     static_cast<int>(calibration_.intrinsics_rgb.sizeY));
  }

  cv::Size2i GetDepthSize() const {
    return cv::Size2i(static_cast<int>(calibration_.intrinsics_d.sizeX),
                      static_cast<int>(calibration_.intrinsics_d.sizeY));
  }

  /// \brief Gets the name of the dataset folder which we are using.
  std::string GetName() const {
    return dataset_folder_.substr(dataset_folder_.rfind('/'));
  }

  DepthProvider* GetDepthProvider() const {
    return depth_provider_;
  }

 private:
  DepthProvider *depth_provider_;

  std::string dataset_folder_;
  int frame_idx_;

  cv::Mat3b left_frame_color_buf_;
  cv::Mat3b right_frame_color_buf_;
  cv::Mat1s depth_buf_;

  // Store the grayscale information necessary for scene flow computation using libviso2, and
  // on-the-fly depth map computation using libelas.
  cv::Mat1b left_frame_gray_buf_;
  cv::Mat1b right_frame_gray_buf_;

  // TODO get rid of this and replace with a similar object which doesn't require ITM.
  ITMLib::Objects::ITMRGBDCalib calibration_;
  StereoCalibration stereo_calibration_;

  std::string GetFrameName(std::string folder, std::string fname_format, int frame_idx) const {
    return folder + "/" + utils::Format(fname_format, frame_idx);
  }
};

} // namespace dynslam

#endif //DYNSLAM_INPUT_H
