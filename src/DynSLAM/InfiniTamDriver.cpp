

#include "InfiniTamDriver.h"

namespace dynslam {
namespace drivers {

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix, bool flip_ud, bool fix_roll) {
  Matrix4f M;
  for(int i = 0; i < 16; ++i) {
    M.m[i] = static_cast<float>(pangolin_matrix.m[i]);
  }

  // This code is convoluted and doesn't actually seem to help improve the UX too much...
  /*
  double yaw = atan2(pangolin_matrix(2, 1), pangolin_matrix(2, 2));
  double pitch = atan2(-pangolin_matrix(2, 0),
                       sqrt(pangolin_matrix(2, 1) * pangolin_matrix(2, 1) +
                            pangolin_matrix(2, 2) * pangolin_matrix(2, 2)));
  double roll = atan2(pangolin_matrix(1, 0), pangolin_matrix(0, 0));

  if (flip_ud) {
    // Fixes the unintuitive behaviour caused by the differences between ITM's raycaster and the
    // default 3D inspector from Pangolin.
//    yaw *= -1.0;
  }

  if (fix_roll) {
    // Prevent the camera from tilting sideways, since it's not useful when inspecting 3D street
    // reconstructions.
//    roll = 0.0;
  }

  Matrix3f new_x;
  new_x(0, 0) = 1; new_x(0, 1) = 0; new_x(0, 2) = 0;
  new_x(1, 0) = 0; new_x(1, 1) = cos(yaw); new_x(1, 2) = -sin(yaw);
  new_x(2, 0) = 0; new_x(2, 1) = sin(yaw); new_x(2, 2) = cos(yaw);

  Matrix3f new_y;
  new_y(0, 0) = cos(pitch); new_y(0, 1) = 0; new_y(0, 2) = sin(pitch);
  new_y(1, 0) = 0; new_y(1, 1) = 1; new_y(1, 2) = 0;
  new_y(2, 0) = -sin(pitch); new_y(2, 1) = 0; new_y(2, 2) = cos(pitch);

  Matrix3f new_z;
  new_z(0, 0) = cos(roll); new_z(0, 1) = -sin(roll); new_z(0, 2) = 0;
  new_z(1, 0) = sin(roll); new_z(1, 1) = cos(roll); new_z(0, 2) = 0;
  new_z(2, 0) = 0; new_z(2, 1) = 0; new_z(2, 2) = 1;
   */

  ITMPose itm_pose;
  itm_pose.SetM(M);
//  itm_pose.SetR(new_z * new_y * new_x);
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
