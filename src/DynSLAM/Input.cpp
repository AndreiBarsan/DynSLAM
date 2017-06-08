
#include "Input.h"

namespace dynslam {

using namespace std;

bool Input::HasMoreImages() {
  string next_fpath = GetFrameName(
      dataset_folder_, config_.left_gray_folder, config_.fname_format,
      frame_idx_);
  return utils::FileExists(next_fpath);
}

bool Input::ReadNextFrame() {
  left_frame_gray_buf_ = cv::imread(GetFrameName(dataset_folder_,
                                                 config_.left_gray_folder,
                                                 config_.fname_format,
                                                 frame_idx_),
                                    CV_LOAD_IMAGE_UNCHANGED);
  right_frame_gray_buf_ = cv::imread(GetFrameName(dataset_folder_,
                                                  config_.right_gray_folder,
                                                  config_.fname_format,
                                                  frame_idx_),
                                     CV_LOAD_IMAGE_UNCHANGED);
  left_frame_color_buf_ = cv::imread(GetFrameName(dataset_folder_,
                                                  config_.left_color_folder,
                                                  config_.fname_format,
                                                  frame_idx_));
  right_frame_color_buf_ = cv::imread(GetFrameName(dataset_folder_,
                                                   config_.right_color_folder,
                                                   config_.fname_format,
                                                   frame_idx_));

  // Sanity checks to ensure the dimensions from the calibration file and the actual image
  // dimensions correspond.
  const auto &rgb_size = GetRgbSize();
  if (left_frame_color_buf_.rows != rgb_size.height ||
      left_frame_color_buf_.cols != rgb_size.width) {
    cerr << "Unexpected left RGB frame size. Got " << left_frame_color_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
    cerr << "Was using format [" << config_.fname_format << "] in dir ["
         << config_.left_color_folder << "]." << endl;
    return false;
  }

  if (right_frame_color_buf_.rows != rgb_size.height ||
      right_frame_color_buf_.cols != rgb_size.width) {
    cerr << "Unexpected right RGB frame size. Got " << right_frame_color_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
    cerr << "Was using format [" << config_.fname_format << "] in dir ["
         << config_.right_color_folder << "]." << endl;
    return false;
  }

  depth_provider_->DepthFromStereo(left_frame_color_buf_,
                                 right_frame_color_buf_,
                                 stereo_calibration_,
                                 depth_buf_);

  const auto &depth_size = GetDepthSize();
  if (depth_buf_.rows != depth_size.height || depth_buf_.cols != depth_size.width) {
    cerr << "Unexpected depth map size. Got " << depth_buf_.size() << ", but the "
         << "calibration file specified " << depth_size << "." << endl;
    cerr << "Was using format [" << config_.depth_fname_format << "] in dir ["
         << config_.depth_folder << "]." << endl;
    return false;
  }

  frame_idx_++;
  return true;
}

void Input::GetCvImages(cv::Mat3b **rgb, cv::Mat1s **raw_depth) {
  *rgb = &left_frame_color_buf_;
  *raw_depth = &depth_buf_;
}

void Input::GetCvStereoGray(cv::Mat1b **left, cv::Mat1b **right) {
  *left = &left_frame_gray_buf_;
  *right = &right_frame_gray_buf_;
}

} // namespace dynslam
