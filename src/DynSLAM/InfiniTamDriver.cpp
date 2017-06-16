

#include "InfiniTamDriver.h"

#include <Eigen/Core>

namespace dynslam {
namespace drivers {

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix) {
  Matrix4f M;
  for(int i = 0; i < 16; ++i) {
    M.m[i] = static_cast<float>(pangolin_matrix.m[i]);
  }

  ITMPose itm_pose;
  itm_pose.SetM(M);
  itm_pose.Coerce();

  return itm_pose;
}

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

Eigen::Matrix4f ItmToEigen(const Matrix4f &itm_matrix) {
  Eigen::Matrix4f res;
  res << itm_matrix.at(0, 0), itm_matrix(1, 0), itm_matrix.at(2, 0), itm_matrix.at(3, 0),
    itm_matrix.at(0, 1), itm_matrix.at(1, 1), itm_matrix.at(2, 1), itm_matrix.at(3, 1),
    itm_matrix.at(0, 2), itm_matrix.at(1, 2), itm_matrix.at(2, 2), itm_matrix.at(3, 2),
    itm_matrix.at(0, 3), itm_matrix.at(1, 3), itm_matrix.at(2, 3), itm_matrix.at(3, 3);
  return res;
}

Matrix4f EigenToItm(const Eigen::Matrix4f &eigen_matrix) {
  return Matrix4f(eigen_matrix(0, 0), eigen_matrix(0, 1), eigen_matrix(0, 2), eigen_matrix(0, 3),
                  eigen_matrix(1, 0), eigen_matrix(1, 1), eigen_matrix(1, 2), eigen_matrix(1, 3),
                  eigen_matrix(2, 0), eigen_matrix(2, 1), eigen_matrix(2, 2), eigen_matrix(2, 3),
                  eigen_matrix(3, 0), eigen_matrix(3, 1), eigen_matrix(3, 2), eigen_matrix(3, 3));
}

void InfiniTamDriver::GetImage(ITMUChar4Image *out,
                               ITMMainEngine::GetImageType get_image_type,
                               const pangolin::OpenGlMatrix &model_view) {
  if (nullptr != this->view) {
    ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    ITMIntrinsics intrinsics = this->view->calib->intrinsics_d;
    ITMMainEngine::GetImage(
        out,
        get_image_type,
        &itm_freeview_pose,
        &intrinsics);
  }
  // Otherwise: We're before the very first frame, so no raycast is available yet.
}

void InfiniTamDriver::UpdateView(const cv::Mat3b &rgb_image,
                                 const cv::Mat_<uint16_t> &raw_depth_image) {
  CvToItm(rgb_image, rgb_itm_);
  CvToItm(raw_depth_image, raw_depth_itm_);

  // * If 'view' is null, this allocates its RGB and depth buffers.
  // * Afterwards, it converts the depth map we give it into a float depth map (we may be able to
  //   skip this step in our case, since we have control over how our depth map is computed).
  // * It then filters the shit out of the depth map (maybe we could skip this?) using five steps
  //   of bilateral filtering.
  this->viewBuilder->UpdateView(&view, rgb_itm_, raw_depth_itm_, settings->useBilateralFilter,
                                settings->modelSensorNoise);
}

} // namespace drivers}
} // namespace dynslam
