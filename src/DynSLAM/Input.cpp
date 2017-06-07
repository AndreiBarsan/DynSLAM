
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

void CvToItm(const cv::Mat3b &mat, ITMUChar4Image *out_itm) {
  Vector2i newSize(mat.cols, mat.rows);
  out_itm->ChangeDims(newSize);
  Vector4u *data_ptr = out_itm->GetData(MEMORYDEVICE_CPU);

  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      int idx = i * mat.cols + j;
      // Convert from OpenCV's standard BGR format to RGB.
      cv::Vec3b col = mat.at<cv::Vec3b>(i, j);
      data_ptr[idx].b = col[0];
      data_ptr[idx].g = col[1];
      data_ptr[idx].r = col[2];
      data_ptr[idx].a = 255u;
    }
  }

  // This does not currently work because the input images lack the alpha channel.
//    memcpy(data_ptr, mat.data, mat.rows * mat.cols * 4 * sizeof(unsigned char));
}

void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_itm) {
  short *data_ptr = out_itm->GetData(MEMORYDEVICE_CPU);
  out_itm->ChangeDims(Vector2i(mat.cols, mat.rows));
  memcpy(data_ptr, mat.data, mat.rows * mat.cols * sizeof(short));
}

void ItmToCv(const ITMUChar4Image &itm, cv::Mat3b *out_mat) {
  // TODO(andrei): Suport resizing the matrix, if necessary.
  const Vector4u *itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  for (int i = 0; i < itm.noDims[1]; ++i) {
    for (int j = 0; j < itm.noDims[0]; ++j) {
      out_mat->at<cv::Vec3b>(i, j) = cv::Vec3b(
          itm_data[i * itm.noDims[0] + j].b,
          itm_data[i * itm.noDims[0] + j].g,
          itm_data[i * itm.noDims[0] + j].r
      );
    }
  }
}

void ItmToCv(const ITMShortImage &itm, cv::Mat1s *out_mat) {
  // TODO(andrei): Suport resizing the matrix, if necessary.
  const int16_t *itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  memcpy(out_mat->data, itm_data, itm.noDims[0] * itm.noDims[1] * sizeof(int16_t));
}

// TODO(andrei): Float -> RGB color image with color map which makes it easier to visualize depth.
void ItmToCv(const ITMFloatImage &itm, cv::Mat1s *out_mat) {
  const float* itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  for (int i = 0; i < itm.noDims[1]; ++i) {
    for (int j = 0; j < itm.noDims[0]; ++j) {
      out_mat->at<int16_t>(i, j) = static_cast<int16_t>(
          itm_data[i * itm.noDims[0] + j] * (numeric_limits<int16_t>::max() / 4)
      );
    }
  }
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

  depth_engine_->DepthFromStereo(left_frame_color_buf_,
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
