
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
  struct Config {
    std::string left_gray_folder;
    std::string right_gray_folder;
    std::string left_color_folder;
    std::string right_color_folder;
    std::string fname_format;
    std::string itm_calibration_fname;

    // These are optional, and only used for precomputed depth/segmentation.
    std::string depth_folder = "";
    std::string depth_fname_format = "";
    // Whether we read direct metric depth from the file, or just disparity values expressed in
    // pixels.
    bool read_depth = false;
    // No format specifier for segmentation information, since the segmented frames' names are based
    // on the RGB frame file names. See `PrecomputedSegmentationProvider` for more information.
    std::string segmentation_folder = "";

    // Whether to read ground truth odometry information from an OxTS dump folder (e.g., KITTI
    // dataset), or from a single-file ground truth, as provided with the kitti-odometry dataset.
    bool odometry_oxts = false;   // TODO(andrei): Support this.
    std::string odometry_fname = "";
  };

  /// We don't use constants here in order to make the code easier to read.
  static Config KittiOdometryConfig() {
    Config config;
    config.left_gray_folder       = "image_0";
    config.right_gray_folder      = "image_1";
    config.left_color_folder      = "image_2";
    config.right_color_folder     = "image_3";
    config.fname_format           = "%06d.png";
    config.itm_calibration_fname  = "itm-calib.txt";

    config.depth_folder           = "precomputed-depth/Frames";
    config.depth_fname_format     = "%04d.pgm";
    config.read_depth             = true;

    config.segmentation_folder    = "seg_image_2/mnc";

    config.odometry_oxts          = false;
    config.odometry_fname         = "ground-truth-poses.txt";

    return config;
  };

  static Config KittiOdometryDispnetConfig() {
    Config config = KittiOdometryConfig();
    config.depth_folder           = "precomputed-depth-dispnet";
    config.depth_fname_format     = "%06d.pfm";
    config.read_depth             = false;
    return config;
  }

 public:
  Input(const std::string &dataset_folder,
        const Config &config,
        DepthProvider *depth_provider,
        const ITMLib::Objects::ITMRGBDCalib &calibration,
        const StereoCalibration &stereo_calibration)
      : dataset_folder_(dataset_folder),
        config_(config),
        depth_provider_(depth_provider),
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
  std::string dataset_folder_;
  Config config_;
  DepthProvider *depth_provider_;
  int frame_idx_;
  // TODO-LOW(andrei): get rid of this and replace with a similar object which doesn't require ITM.
  ITMLib::Objects::ITMRGBDCalib calibration_;
  StereoCalibration stereo_calibration_;

  cv::Mat3b left_frame_color_buf_;
  cv::Mat3b right_frame_color_buf_;
  cv::Mat1s depth_buf_;

  // Store the grayscale information necessary for scene flow computation using libviso2, and
  // on-the-fly depth map computation using libelas.
  cv::Mat1b left_frame_gray_buf_;
  cv::Mat1b right_frame_gray_buf_;

  static std::string GetFrameName(std::string root,
                                  std::string folder,
                                  std::string fname_format,
                                  int frame_idx) {
    return root + "/" + folder + "/" + utils::Format(fname_format, frame_idx);
  }
};

} // namespace dynslam

#endif //DYNSLAM_INPUT_H
