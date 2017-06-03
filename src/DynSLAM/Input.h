
#ifndef DYNSLAM_INPUT_H
#define DYNSLAM_INPUT_H

#include <string>
#include <highgui.h>

#include "DepthEngine.h"
#include "Utils.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Utils/ITMLibDefines.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMRGBDCalib.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Utils/ITMCalibIO.h"
#include "../InfiniTAM/InfiniTAM/Utils/FileUtils.h"

namespace dynslam {

/// \brief Converts an OpenCV RGB Mat into an InfiniTAM image.
void CvToItm(const cv::Mat &mat, ITMUChar4Image *out_rgb);

/// \brief Converts an OpenCV depth Mat into an InfiniTAM depth image.
void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_depth);

// TODO do not depend on infinitam objects. The ITM driver should be the only bit worrying about
// them.
ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const std::string& fpath);

// TODO(andrei): Better name and docs for this once the interface is fleshed out.
// TODO(andrei): Move code to cpp.
class Input {
 public:
  Input(const std::string &dataset_folder,
        DepthEngine *depth_engine,
        const ITMLib::Objects::ITMRGBDCalib &calibration)
      : depth_engine_(depth_engine),
        dataset_folder_(dataset_folder),
        frame_idx_(0),
        calibration_(calibration) {}

  bool HasMoreImages();

  // TODO get rid of this and use some other format
  /// \brief Reads from the input folders into the specified InfiniTAM buffers.
  /// \return True if the images could be loaded and processed appropriately.
  bool GetITMImages(ITMUChar4Image *rgb, ITMShortImage *raw_depth);

  ITMLib::Objects::ITMRGBDCalib GetITMCalibration() {
    std::cerr << "Warning: Using deprecated ITM calibration accessor!" << std::endl;
    return calibration_;
  };

  cv::Size2i GetRgbSize() {
    return cv::Size2i(static_cast<int>(calibration_.intrinsics_rgb.sizeX),
                     static_cast<int>(calibration_.intrinsics_rgb.sizeY));
  }

  cv::Size2i GetDepthSize() {
    return cv::Size2i(static_cast<int>(calibration_.intrinsics_d.sizeX),
                      static_cast<int>(calibration_.intrinsics_d.sizeY));
  }

  std::string GetName() {
    return dataset_folder_.substr(dataset_folder_.rfind('/'));
  }

 private:
  DepthEngine *depth_engine_;

  std::string dataset_folder_;
  int frame_idx_;

  cv::Mat left_frame_buf_;
  cv::Mat right_frame_buf_;
  cv::Mat_<uint16_t> depth_buf_;
//  cv::Mat disparity_buf_;

  // TODO get rid of this
  ITMLib::Objects::ITMRGBDCalib calibration_;

  // TODO dedicated subclass for reading stereo input
  std::string GetRgbFrameName(std::string folder, std::string fname_format, int frame_idx) const {
    return folder + "/" + utils::Format(fname_format, frame_idx);
  }

};

} // namespace dynslam

#endif //DYNSLAM_INPUT_H
