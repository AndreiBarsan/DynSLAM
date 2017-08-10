

#include "InfiniTamDriver.h"

// TODO(andrei): Why not move to DynSLAM.cpp?
DEFINE_bool(enable_evaluation, true, "Whether to enable evaluation mode for DynSLAM. This means "
    "the system will load in LIDAR ground truth and compare its maps with it, dumping the results "
    "in CSV format.");

namespace dynslam {
namespace drivers {

using namespace dynslam::utils;

/// \brief Converts between the DynSlam preview type enums and the InfiniTAM ones.
ITMMainEngine::GetImageType GetItmVisualization(PreviewType preview_type) {
  switch(preview_type) {
    case PreviewType::kDepth:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_DEPTH;
    case PreviewType::kGray:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_SHADED;
    case PreviewType::kColor:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME;
    case PreviewType::kNormal:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL;
    case PreviewType::kWeight:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH_WEIGHT;
    case PreviewType::kLatestRaycast:
      return ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST;

    default:
      throw runtime_error(Format("Unknown preview type: %d", preview_type));
  }
}

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

ITMLib::Objects::ITMRGBDCalib *CreateItmCalib(const Eigen::Matrix<double, 3, 4> &left_cam_proj,
                                              const Eigen::Vector2i &frame_size) {
  ITMRGBDCalib *calib = new ITMRGBDCalib;
  float kMetersToMillimeters = 1.0f / 1000.0f;

  ITMIntrinsics intrinsics;
  float fx = static_cast<float>(left_cam_proj(0, 0));
  float fy = static_cast<float>(left_cam_proj(1, 1));
  float cx = static_cast<float>(left_cam_proj(0, 2));
  float cy = static_cast<float>(left_cam_proj(1, 2));
  float sizeX = frame_size(0);
  float sizeY = frame_size(1);
  intrinsics.SetFrom(fx, fy, cx, cy, sizeX, sizeY);

  // Our intrinsics are always the same for RGB and depth since we compute depth from stereo.
  // TODO-LOW(andrei): But not if we compute the depth map from the gray camera frames, which have a
  // subtle-but-non-negligible offset from the color ones. Note that for now we're simply computing
  // all depth maps using the RGB inputs.
  calib->intrinsics_rgb = intrinsics;
  calib->intrinsics_d = intrinsics;

  // RGB and depth "sensors" are one and the same, so the relative pose is the identity matrix.
  // TODO-LOW(andrei): Not necessarily. See above.
  Matrix4f identity; identity.setIdentity();
  calib->trafo_rgb_to_depth.SetFrom(identity);

  // These parameters are used by ITM to convert from the input depth, expressed in millimeters, to
  // the internal depth, which is expressed in meters.
  calib->disparityCalib.SetFrom(kMetersToMillimeters, 0.0f, ITMDisparityCalib::TRAFO_AFFINE);
  return calib;
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

void FloatDepthmapToShort(const float *pixels, cv::Mat1s &out_mat) {
  const int kMetersToMillimeters = 1000;
  for (int i = 0; i < out_mat.rows; ++i) {
    for (int j = 0; j < out_mat.cols; ++j) {
      /// ITM internal: depth = meters, float
      /// Our preview:  depth = mm, short int
      out_mat.at<int16_t>(i, j) = static_cast<int16_t>(
          pixels[i * out_mat.cols + j] * kMetersToMillimeters
      );
    }
  }
}

void ItmDepthToCv(const ITMFloatImage &itm, cv::Mat1s *out_mat) {
  const float* itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  FloatDepthmapToShort(itm_data, *out_mat);
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
  // Note the ordering, which is necessary since the input to the ITM matrix must be given in
  // column-major format.
  Matrix4f res(eigen_matrix(0, 0), eigen_matrix(1, 0), eigen_matrix(2, 0), eigen_matrix(3, 0),
               eigen_matrix(0, 1), eigen_matrix(1, 1), eigen_matrix(2, 1), eigen_matrix(3, 1),
               eigen_matrix(0, 2), eigen_matrix(1, 2), eigen_matrix(2, 2), eigen_matrix(3, 2),
               eigen_matrix(0, 3), eigen_matrix(1, 3), eigen_matrix(2, 3), eigen_matrix(3, 3));
  return res;
}

void InfiniTamDriver::GetImage(ITMUChar4Image *out,
                               dynslam::PreviewType get_image_type,
                               const pangolin::OpenGlMatrix &model_view) {
  if (nullptr != this->view) {
    ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    if (get_image_type == PreviewType::kDepth) {
      // TODO(andrei): Make this possible again.
      cerr << "Warning: Cannot preview depth normally anymore." << endl;
      return;
    }

    ITMIntrinsics intrinsics = this->viewBuilder->GetCalib()->intrinsics_d;
    ITMMainEngine::GetImage(
        out,
        nullptr,
        GetItmVisualization(get_image_type),
        &itm_freeview_pose,
        &intrinsics);
  }
  // Otherwise: We're before the very first frame, so no raycast is available yet.
}

void InfiniTamDriver::GetFloatImage(
    ITMFloatImage *out,
    dynslam::PreviewType get_image_type,
    const pangolin::OpenGlMatrix &model_view
) {
  if (nullptr != this->view) {
    ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    if (get_image_type != PreviewType::kDepth) {
      cerr << "Warning: Can only preview depth as float." << endl;
      return;
    }

    ITMIntrinsics intrinsics = this->viewBuilder->GetCalib()->intrinsics_d;
    ITMMainEngine::GetImage(
        nullptr,
        out,
        GetItmVisualization(get_image_type),
        &itm_freeview_pose,
        &intrinsics);
  }
}

void InfiniTamDriver::UpdateView(const cv::Mat3b &rgb_image,
                                 const cv::Mat1s &raw_depth_image) {
  CvToItm(rgb_image, rgb_itm_);
  CvToItm(raw_depth_image, raw_depth_itm_);

  // * If 'view' is null, this allocates its RGB and depth buffers.
  // * Afterwards, it converts the depth map we give it into a float depth map (we may be able to
  //   skip this step in our case, since we have control over how our depth map is computed).
  // * It then filters the shit out of the depth map (maybe we could skip this?) using five steps
  //   of bilateral filtering.
  // * Note: ITM internally uses ITMShortImages, so SIGNED short.
  this->viewBuilder->UpdateView(&view, rgb_itm_, raw_depth_itm_, settings->useBilateralFilter,
                                settings->modelSensorNoise);
}

} // namespace drivers
} // namespace dynslam
