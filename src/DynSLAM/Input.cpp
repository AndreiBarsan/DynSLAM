
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
  string next_fpath = GetRgbFrameName(
      dataset_folder_ + "/image_0/",
      "%06d.png",
      frame_idx_);
  return utils::file_exists(next_fpath);
}

bool Input::GetITMImages(ITMUChar4Image *rgb, ITMShortImage *raw_depth) {
  string left_folder = dataset_folder_ + "/image_2";
  string right_folder = dataset_folder_ + "/image_3";
  string rgb_frame_fname_format = "%06d.png";
  string left_frame_fpath = GetRgbFrameName(left_folder, rgb_frame_fname_format, frame_idx_);
  string right_frame_fpath = GetRgbFrameName(right_folder, rgb_frame_fname_format, frame_idx_);
  left_frame_buf_ = cv::imread(left_frame_fpath);
  right_frame_buf_ = cv::imread(right_frame_fpath);

  // Sanity checks to ensure the dimensions from the calibration file and the actual image
  // dimensions correspond.
  const auto &rgb_size = GetRgbSize();
  if (left_frame_buf_.rows != rgb_size.height || left_frame_buf_.cols != rgb_size.width) {
    cerr << "Unexpected left RGB frame size. Got " << left_frame_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
    return false;
  }

  if (right_frame_buf_.rows != rgb_size.height || right_frame_buf_.cols != rgb_size.width) {
    cerr << "Unexpected right RGB frame size. Got " << right_frame_buf_.size() << ", but the "
        << "calibration file specified " << rgb_size << "." << endl;
    return false;
  }

  // The left frame is the RGB input to our system.
  CvToItm(left_frame_buf_, rgb);

  depth_engine_->DisparityMapFromStereo(left_frame_buf_, right_frame_buf_, depth_buf_);

  const auto &depth_size = GetDepthSize();
  if (depth_buf_.rows != depth_size.height || depth_buf_.cols != depth_size.width) {
    cerr << "Unexpected depth map size. Got " << depth_buf_.size() << ", but the "
         << "calibration file specified " << depth_size << "." << endl;
    return false;
  }

  // TODO(andrei): Make sure you actually use this. ATM, libelas-tooling's kitti2klg does the
  // depth from disparity calculation!
//    StereoCalibration stereo_calibration(0, 0);
//    depth_engine_->DepthFromDisparityMap(disparity, stereo_calibration, depth);
  CvToItm(depth_buf_, raw_depth);

  frame_idx_++;
  return true;
}

} // namespace dynslam
