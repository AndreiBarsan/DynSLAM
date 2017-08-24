
#ifndef DYNSLAM_INPUT_H
#define DYNSLAM_INPUT_H

#include <string>
#include <highgui.h>
#include <memory>

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
    std::string dataset_name;
    std::string left_gray_folder;
    std::string right_gray_folder;
    std::string left_color_folder;
    std::string right_color_folder;
    std::string fname_format;
    std::string calibration_fname;

    /// \brief Minimum depth to keep when computing depth maps.
    float min_depth_m = -1.0f;
    /// \brief Maximum depth to keep when computing depth maps.
    float max_depth_m = -1.0f;

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
    // !! UNSUPPORTED AT THE MOMENT !!
    bool odometry_oxts = false;
    std::string odometry_fname = "";

    /// \brief The Velodyne LIDAR data (used only for evaluation).
    std::string velodyne_folder = "";
    std::string velodyne_fname_format = "";

    /// \brief Tracklet ground truth, only available in the KITTI-tracking benchmark in the format
    ///        that we support.
    std::string tracklet_folder = "";
  };

  /// We don't define the configs as constants here in order to make the code easier to read.
  /// Details and downloads: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
  static Config KittiOdometryConfig() {
    Config config;
    config.dataset_name           = "kitti-odometry";
    config.left_gray_folder       = "image_0";
    config.right_gray_folder      = "image_1";
    config.left_color_folder      = "image_2";
    config.right_color_folder     = "image_3";
    config.fname_format           = "%06d.png";
    config.calibration_fname      = "calib.txt";

    config.min_depth_m            =  0.5f;
    config.max_depth_m            = 20.0f;
    config.depth_folder           = "precomputed-depth/Frames";
    config.depth_fname_format     = "%04d.xml";
    config.read_depth             = true;

    config.segmentation_folder    = "seg_image_2/mnc";

    config.odometry_oxts          = false;
    config.odometry_fname         = "ground-truth-poses.txt";

    config.velodyne_folder        = "velodyne";
    config.velodyne_fname_format  = "%06d.bin";

    return config;
  };

  /// The structure of the tracking dataset is a bit different, and there's no one folder per, so we
  /// must explicitly specify the sequence number (ID).
  /// WARNING: no gray data available for the tracking benchmark sequences.
  /// Details and downloads: http://www.cvlibs.net/datasets/kitti/eval_tracking.php
  static Config KittiTrackingConfig(int sequence_id) {
    Config config;
    config.dataset_name           = utils::Format("kitti-tracking-sequence-%04d", sequence_id);

    config.left_gray_folder       = utils::Format("training/image_02/%04d/", sequence_id);
    config.right_gray_folder      = utils::Format("training/image_03/%04d/", sequence_id);
    config.left_color_folder      = utils::Format("training/image_02/%04d/", sequence_id);
    config.right_color_folder     = utils::Format("training/image_03/%04d/", sequence_id);
    config.fname_format           = "%06d.png";
    config.calibration_fname      = utils::Format("training/calib/%04d.txt", sequence_id);

    config.min_depth_m = 0.5f;
    config.max_depth_m = 20.0f;
    config.depth_folder           = utils::Format("training/precomputed-depth/%04d/Frames", sequence_id);
    config.depth_fname_format     = "%04d.xml";
    config.read_depth = true;
    config.segmentation_folder    = utils::Format("training/seg_image_02/%04d/mnc", sequence_id);

    config.odometry_oxts          = false;
    config.odometry_fname         = "";

    config.velodyne_folder        = utils::Format("training/velodyne/%04d/", sequence_id);
    config.velodyne_fname_format  = "%06d.bin";

    config.tracklet_folder        = utils::Format("training/label_02/%04d.txt", sequence_id);
    return config;
  }

  static Config KittiTrackingDispnetConfig(int sequence_id) {
    Config config                 = KittiTrackingConfig(sequence_id);
    config.depth_folder           = utils::Format("training/precomputed-depth-dispnet/%04d", sequence_id);
    config.depth_fname_format     = "%06d.pfm";
    config.read_depth             = false;
    return config;
  }

  static Config KittiOdometryLowresConfig(float factor) {
    Config config = KittiOdometryConfig();
    config.left_gray_folder       = utils::Format("image_0_%.2f", factor);
    config.right_gray_folder      = utils::Format("image_1_%.2f", factor);
    config.left_color_folder      = utils::Format("image_2_%.2f", factor);
    config.right_color_folder     = utils::Format("image_3_%.2f", factor);

    config.depth_folder           = utils::Format("precomputed-depth-elas-%.2f/Frames", factor);
    config.segmentation_folder    = utils::Format("seg_image_2-%.2f/mnc", factor);

    return config;
  }

  static Config KittiOdometryDispnetConfig() {
    Config config                 = KittiOdometryConfig();
    config.depth_folder           = "precomputed-depth-dispnet";
    config.depth_fname_format     = "%06d.pfm";
    config.read_depth             = false;
    return config;
  }

  static Config KittiOdometryDispnetLowresConfig(float factor) {
    Config config = KittiOdometryDispnetConfig();
    config.left_gray_folder       = utils::Format("image_0_%.2f", factor);
    config.right_gray_folder      = utils::Format("image_1_%.2f", factor);
    config.left_color_folder      = utils::Format("image_2_%.2f", factor);
    config.right_color_folder     = utils::Format("image_3_%.2f", factor);

    config.depth_folder           = utils::Format("precomputed-depth-dispnet-%.2f", factor);
    config.segmentation_folder    = utils::Format("seg_image_2-%.2f/mnc", factor);

    return config;
  }


 public:
  Input(const std::string &dataset_folder,
        const Config &config,
        DepthProvider *depth_provider,
        const Eigen::Vector2i &frame_size,
        const StereoCalibration &stereo_calibration,
        int frame_offset,
        float input_scale)
      : dataset_folder_(dataset_folder),
        config_(config),
        depth_provider_(depth_provider),
        frame_offset_(frame_offset),
        frame_idx_(frame_offset),
        frame_width_(frame_size(0)),
        frame_height_(frame_size(1)),
        stereo_calibration_(stereo_calibration),
        depth_buf_(frame_size(1), frame_size(0)),
        input_scale_(input_scale),
        depth_buf_small_(static_cast<int>(round(frame_size(1) * input_scale)),
                         static_cast<int>(round(frame_size(0) * input_scale)))
  {}

  bool HasMoreImages() const;

  /// \brief Advances the input reader to the next frame.
  /// \returns True if the next frame's files could be read successfully.
  bool ReadNextFrame();

  /// \brief Returns pointers to the latest RGB and depth data.
  /// \note The caller does not take ownership.
  void GetCvImages(cv::Mat3b **rgb, cv::Mat1s **raw_depth);

  /// \brief Returns pointers to the latest grayscale input frames.
  void GetCvStereoGray(cv::Mat1b **left, cv::Mat1b **right);

  /// \brief Returns pointers to the latest color input frames.
  void GetCvStereoColor(cv::Mat3b **left_rgb, cv::Mat3b **right_rgb);

  cv::Size2i GetRgbSize() const {
    return cv::Size2i(frame_width_, frame_height_);
  }

  cv::Size2i GetDepthSize() const {
    return cv::Size2i(frame_width_, frame_height_);
  }

  /// \brief Gets the name of the dataset folder which we are using.
  /// TODO(andrei): Make this more robust.
  std::string GetSequenceName() const {
    return dataset_folder_.substr(dataset_folder_.rfind('/') + 1);
  }

  std::string GetDatasetIdentifier() const {
    return config_.dataset_name + "-" + GetSequenceName();
  }

  DepthProvider* GetDepthProvider() const {
    return depth_provider_;
  }

  void SetDepthProvider(DepthProvider *depth_provider) {
    this->depth_provider_ = depth_provider;
  }

  /// \brief Returns the current frame index from the dataset.
  /// \note May not be the same as the current DynSLAM frame number if an offset was used.
  int GetCurrentFrame() const {
    return frame_idx_;
  }

  /// \brief Sets the out parameters to the RGB and depth images from the specified frame.
  void GetFrameCvImages(int frame_idx, std::shared_ptr<cv::Mat3b> &rgb, std::shared_ptr<cv::Mat1s> &raw_depth);

  const Config& GetConfig() const {
    return config_;
  }

  int GetFrameOffset() const {
    return frame_offset_;
  }

 private:
  std::string dataset_folder_;
  Config config_;
  DepthProvider *depth_provider_;
  const int frame_offset_;
  int frame_idx_;
  int frame_width_;
  int frame_height_;

  StereoCalibration stereo_calibration_;

  cv::Mat3b left_frame_color_buf_;
  cv::Mat3b right_frame_color_buf_;
  cv::Mat1s depth_buf_;

  // Store the grayscale information necessary for scene flow computation using libviso2, and
  // on-the-fly depth map computation using libelas.
  cv::Mat1b left_frame_gray_buf_;
  cv::Mat1b right_frame_gray_buf_;

  /// Used when evaluating low-resolution input.
  float input_scale_;
  cv::Mat1s depth_buf_small_;
//  cv::Mat1s raw_depth_small(static_cast<int>(round(GetDepthSize().height * input_scale_)),
//  static_cast<int>(round(GetDepthSize().width * input_scale_)));

  static std::string GetFrameName(const std::string &root,
                                  const std::string &folder,
                                  const std::string &fname_format,
                                  int frame_idx) {
    return root + "/" + folder + "/" + utils::Format(fname_format, frame_idx);
  }

  void ReadLeftGray(int frame_idx, cv::Mat1b &out) const;
  void ReadRightGray(int frame_idx, cv::Mat1b &out) const;
  void ReadLeftColor(int frame_idx, cv::Mat3b &out) const;
  void ReadRightColor(int frame_idx, cv::Mat3b &out) const;
};

} // namespace dynslam

#endif //DYNSLAM_INPUT_H
