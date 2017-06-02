#ifndef DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
#define DYNSLAM_PRECOMPUTEDDEPTHENGINE_H

#include <string>

#include "DepthEngine.h"

namespace dynslam {

/// \brief Reads precomputed depth maps from a folder.
/// The depth maps are expected to be grayscale, and in 8 or 16-bit format.
class PrecomputedDepthEngine : public DepthEngine {
 public:
  virtual ~PrecomputedDepthEngine() {}

  PrecomputedDepthEngine(const std::string &folder, const std::string &fname_format)
      : folder(folder),
        fname_format(fname_format),
        frame_idx(0) {}

  virtual void DisparityMapFromStereo(const cv::Mat &left,
                                      const cv::Mat &right,
                                      cv::Mat &out_disparity) override;

  virtual float DepthFromDisparity(const float disparity_px,
                                   const StereoCalibration &calibration) override;

 private:
  std::string folder;
  /// \brief The printf-style format of the frame filenames, such as "frame-%04d.png" for frames
  /// which are called "frame-0000.png"-"frame-9999.png".
  std::string fname_format;
  int frame_idx;
};

} // namespace dynslam

#endif //DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
