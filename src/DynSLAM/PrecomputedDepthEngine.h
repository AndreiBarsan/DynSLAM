#ifndef DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
#define DYNSLAM_PRECOMPUTEDDEPTHENGINE_H

#include <string>

#include "DepthEngine.h"

namespace dynslam {

/// \brief Reads precomputed disparity (default) or depth maps from a folder.
/// The depth maps are expected to be grayscale, and in 8 or 16-bit format if 'hdr_depth_' if
/// false, and in 32-bit PFM format otherwise.
class PrecomputedDepthEngine : public DepthEngine {
 public:
  virtual ~PrecomputedDepthEngine() {}

  // TODO(andrei): Do we still need the HDR flag?
  PrecomputedDepthEngine(const std::string &folder, const std::string &fname_format,
                         bool hdr_depth = false, bool input_is_depth = false)
      : DepthEngine(input_is_depth),
        folder(folder),
        fname_format(fname_format),
        frame_idx(0),
        hdr_depth_(hdr_depth) {}

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

  /// \brief Whether the depth is in HDR format (32-bit float). Otherwise, assumes 16-bit integers.
  bool hdr_depth_;

};

} // namespace dynslam

#endif //DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
