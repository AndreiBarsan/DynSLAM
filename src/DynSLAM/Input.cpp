
#include "Input.h"

namespace dynslam {

using namespace std;

bool Input::HasMoreImages() {
  string next_fpath = GetFrameName(
      dataset_folder_ + "/image_0/",
      "%06d.png",
      frame_idx_);
  return utils::FileExists(next_fpath);
}

bool Input::ReadNextFrame() {
  string left_gray_folder = dataset_folder_ + "/image_0";
  string right_gray_folder = dataset_folder_ + "/image_1";
  string left_color_folder = dataset_folder_ + "/image_2";
  string right_color_folder = dataset_folder_ + "/image_3";
  string fname_format = "%06d.png";

  left_frame_gray_buf_ = cv::imread(GetFrameName(left_gray_folder, fname_format, frame_idx_),
                                    CV_LOAD_IMAGE_UNCHANGED);
  right_frame_gray_buf_ = cv::imread(GetFrameName(right_gray_folder, fname_format, frame_idx_),
                                     CV_LOAD_IMAGE_UNCHANGED);
  left_frame_color_buf_ = cv::imread(GetFrameName(left_color_folder, fname_format, frame_idx_));
  right_frame_color_buf_ = cv::imread(GetFrameName(right_color_folder, fname_format, frame_idx_));

  // Sanity checks to ensure the dimensions from the calibration file and the actual image
  // dimensions correspond.
  const auto &rgb_size = GetRgbSize();
  if (left_frame_color_buf_.rows != rgb_size.height ||
      left_frame_color_buf_.cols != rgb_size.width) {
    cerr << "Unexpected left RGB frame size. Got " << left_frame_color_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
    return false;
  }

  if (right_frame_color_buf_.rows != rgb_size.height ||
      right_frame_color_buf_.cols != rgb_size.width) {
    cerr << "Unexpected right RGB frame size. Got " << right_frame_color_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
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
