
#include "Input.h"

namespace dynslam {

using namespace std;

ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const string &fpath) {
  ITMLib::Objects::ITMRGBDCalib out_calib;
  if (! ITMLib::Objects::readRGBDCalib(fpath.c_str(), out_calib)) {
    throw runtime_error(dynslam::utils::Format(
        "Could not read calibration file: [%s]\n", fpath.c_str()));
  }
  return out_calib;
}

void CvToItm(const cv::Mat &mat, ITMUChar4Image *out_rgb) {
  Vector2i newSize(mat.cols, mat.rows);
  out_rgb->ChangeDims(newSize);
  Vector4u *data_ptr = out_rgb->GetData(MEMORYDEVICE_CPU);

  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      int idx = i * mat.cols + j;
      // Convert from OpenCV's standard BGR format to RGB.
      data_ptr[idx].r = mat.data[idx * 3 + 2];
      data_ptr[idx].g = mat.data[idx * 3 + 1];
      data_ptr[idx].b = mat.data[idx * 3 + 0];
      data_ptr[idx].a = 255u;
    }
  }

  // This does not currently work because the input images lack the alpha channel.
//    memcpy(data_ptr, mat.data, mat.rows * mat.cols * 4 * sizeof(unsigned char));
}

void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_depth) {
  short *data_ptr = out_depth->GetData(MEMORYDEVICE_CPU);
  out_depth->ChangeDims(Vector2i(mat.cols, mat.rows));
  memcpy(data_ptr, mat.data, mat.rows * mat.cols * sizeof(short));
}

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

  left_frame_gray_buf_ = cv::imread(GetFrameName(left_gray_folder, fname_format, frame_idx_));
  right_frame_gray_buf_ = cv::imread(GetFrameName(right_gray_folder, fname_format, frame_idx_));
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

  // Parameters used in the KITTI-odometry dataset.
  // TODO(andrei): Pass these things to the Input class!
  float baseline_m = 0.537150654273f;
  float focal_length_px = 707.0912f;
  StereoCalibration stereo_calibration(baseline_m, focal_length_px);
  depth_engine_->DepthFromStereo(left_frame_color_buf_, right_frame_color_buf_, stereo_calibration, depth_buf_);

  const auto &depth_size = GetDepthSize();
  if (depth_buf_.rows != depth_size.height || depth_buf_.cols != depth_size.width) {
    cerr << "Unexpected depth map size. Got " << depth_buf_.size() << ", but the "
         << "calibration file specified " << depth_size << "." << endl;
    return false;
  }

  frame_idx_++;
  return true;
}

void Input::GetItmImages(ITMUChar4Image *rgb, ITMShortImage *raw_depth) {
  // The left frame is the RGB input to our system.
  CvToItm(left_frame_color_buf_, rgb);
  CvToItm(depth_buf_, raw_depth);
}

void Input::GetCvImages(cv::Mat4b &rgb, cv::Mat_<uint16_t> &raw_depth) {
  rgb = left_frame_color_buf_;
  raw_depth = depth_buf_;
}

void Input::GetCvStereoGray(const cv::Mat1b **left, const cv::Mat1b **right) {
  *left = &left_frame_gray_buf_;
  *right = &right_frame_gray_buf_;
}

} // namespace dynslam
