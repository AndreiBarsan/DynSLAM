#ifndef DYNSLAM_PRECOMPUTEDDEPTHPROVIDER_H
#define DYNSLAM_PRECOMPUTEDDEPTHPROVIDER_H

#include <string>

#include "DepthProvider.h"
#include "Input.h"

namespace dynslam {

extern const std::string kDispNetName;
extern const std::string kPrecomputedElas;

/// \brief Reads precomputed disparity (default) or depth maps from a folder.
/// The depth maps are expected to be grayscale, and in short 16-bit or float 32-bit format.
class PrecomputedDepthProvider : public DepthProvider {
 public:
  PrecomputedDepthProvider(
      Input *input,
      const std::string &folder,
      const std::string &fname_format,
      bool input_is_depth,
      int frame_offset,
      float min_depth_m,
      float max_depth_m)
      : DepthProvider(input_is_depth, min_depth_m, max_depth_m),
        input_(input),
        folder_(folder),
        fname_format_(fname_format) {}

  PrecomputedDepthProvider(const PrecomputedDepthProvider&) = delete;
  PrecomputedDepthProvider(PrecomputedDepthProvider&&) = delete;
  PrecomputedDepthProvider& operator=(const PrecomputedDepthProvider&) = delete;
  PrecomputedDepthProvider& operator=(PrecomputedDepthProvider&&) = delete;

  ~PrecomputedDepthProvider() override = default;

  void DisparityMapFromStereo(const cv::Mat &left,
                              const cv::Mat &right,
                              cv::Mat &out_disparity) override;


  /// \brief Loads the precomputed depth map for the specified frame into 'out_depth'.
  void GetDepth(int frame_idx, StereoCalibration &calibration, cv::Mat1s &out_depth, float scale) {
    if (input_is_depth_) {
      std::cout << "Will read precomputed depth..." << std::endl;
      ReadPrecomputed(frame_idx, out_depth);
      std::cout << "Done reading precomputed depth for specific frame [" << frame_idx << "]." << std::endl;
      return;
    }

    ReadPrecomputed(frame_idx, out_disparity_);

    // TODO(andrei): Remove code duplication between this and 'DepthProvider'.
    if (out_disparity_.type() == CV_32FC1) {
      DepthFromDisparityMap<float>(out_disparity_, calibration, out_depth, scale);
    } else if (out_disparity_.type() == CV_16SC1) {
      throw std::runtime_error("Unsupported.");
//      DepthFromDisparityMap<uint16_t>(out_disparity_, calibration, out_depth);
    } else {
      throw std::runtime_error(utils::Format(
          "Unknown data type for disparity matrix [%s]. Supported are CV_32FC1 and CV_16SC1.",
          utils::Type2Str(out_disparity_.type()).c_str()
      ));
    }
  }

  float DepthFromDisparity(const float disparity_px, const StereoCalibration &calibration) override;

  const std::string &GetName() const override;

 protected:
  /// \brief Reads a disparity or depth (depending on the data).
  void ReadPrecomputed(int frame_idx, cv::Mat &out) const;

 private:
  Input *input_;
  std::string folder_;
  /// \brief The printf-style format of the frame filenames, such as "frame-%04d.png" for frames
  /// which are called "frame-0000.png"-"frame-9999.png".
  std::string fname_format_;
};

} // namespace dynslam

#endif //DYNSLAM_PRECOMPUTEDDEPTHPROVIDER_H
