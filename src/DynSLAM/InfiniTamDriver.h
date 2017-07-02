

#ifndef DYNSLAM_INFINITAMDRIVER_H
#define DYNSLAM_INFINITAMDRIVER_H

#include <iostream>

#include <opencv/cv.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>

#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"
#include "Input.h"
#include "PreviewType.h"

namespace dynslam {
namespace drivers {

// Utilities for converting from OpenCV to InfiniTAM vectors.
template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Vec<T, 2> in) {
  return ORUtils::Vector2<T>(in[0], in[1]);
}

template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Size_<T> in) {
  return ORUtils::Vector2<T>(in.width, in.height);
}

template<typename T>
ORUtils::Vector3<T> ToItmVec(const cv::Vec<T, 3> in) {
  return ORUtils::Vector3<T>(in[0], in[1], in[2]);
}

template<typename T>
ORUtils::Vector4<T> ToItmVec(const cv::Vec<T, 4> in) {
  return ORUtils::Vector4<T>(in[0], in[1], in[2], in[3]);
}

// TODO do not depend on infinitam objects. The ITM driver should be the only bit worrying about
// them.
ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const std::string& fpath);

// TODO(andrei): Make */& more consistent.
ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const string &fpath);

/// \brief Converts an OpenCV RGB Mat into an InfiniTAM image.
void CvToItm(const cv::Mat3b &mat, ITMUChar4Image *out_itm);

/// \brief Converts an OpenCV depth Mat into an InfiniTAM depth image.
void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_itm);

/// \brief Converts an InfiniTAM rgb(a) image into an OpenCV RGB mat, discarding the alpha information.
void ItmToCv(const ITMUChar4Image &itm, cv::Mat3b *out_mat);

/// \brief Converts an InfiniTAM depth image into an OpenCV mat.
void ItmToCv(const ITMShortImage &itm, cv::Mat1s *out_mat);

/// \brief Converts an InfiniTAM float depth image into an OpenCV mat.
void ItmToCv(const ITMFloatImage &itm, cv::Mat1s *out_mat);

/// \brief Converts an InfiniTAM 4x4 matrix to an Eigen object.
Eigen::Matrix4f ItmToEigen(const Matrix4f &itm_matrix);

Matrix4f EigenToItm(const Eigen::Matrix4f &eigen_matrix);

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix);

/// \brief Interfaces between DynSLAM and InfiniTAM.
class InfiniTamDriver : public ITMMainEngine {
public:
  // TODO-LOW(andrei): We may need to add another layer of abstraction above the drivers to get the
  // best modularity possible (driver+input+custom settings combos).

  InfiniTamDriver(
      const ITMLibSettings *settings,
      const ITMRGBDCalib *calib,
      const Vector2i &img_size_rgb,
      const Vector2i &img_size_d)
      : ITMMainEngine(settings, calib, img_size_rgb, img_size_d),
        rgb_itm_(new ITMUChar4Image(img_size_rgb, true, true)),
        raw_depth_itm_(new ITMShortImage(img_size_d, true, true)),
        rgb_cv_(new cv::Mat3b(img_size_rgb.height, img_size_rgb.width)),
        raw_depth_cv_(new cv::Mat1s(img_size_d.height, img_size_d.width)),
        last_egomotion_(new Eigen::Matrix4f)
  {
    last_egomotion_->setIdentity();
  }

  virtual ~InfiniTamDriver() {
    delete rgb_itm_;
    delete raw_depth_itm_;
    delete rgb_cv_;
    delete raw_depth_cv_;
    delete last_egomotion_;
  }

  // TODO(andrei): I was passing a Mat4b as rgb, which caused a nasty bug, but didn't even get a
  // warning. Is that normal? I was expecting a little more type safety...
  void UpdateView(const cv::Mat3b &rgb_image, const cv::Mat_<uint16_t> &raw_depth_image);

  // used by the instance reconstruction
  void SetView(ITMView *view) {
    if (this->view) {
      // TODO(andrei): These views should be memory managed by the tracker. Make sure this is right.
//      delete this->view;
    }

    this->view = view;
  }

  void Track() {
    // TODO(andrei): Compute the latest relative motion in a more direct manner.
    Matrix4f old_pose = this->trackingState->pose_d->GetInvM();
    Matrix4f old_pose_inv;
    old_pose.inv(old_pose_inv);
    this->trackingController->Track(this->trackingState, this->view);

    Matrix4f new_pose = this->trackingState->pose_d->GetInvM();
    *(this->last_egomotion_) = ItmToEigen(old_pose_inv * new_pose);
  }

  // TODO(andrei): Document better.
  // Use this to explicitly set tracking state, e.g., when reconstructing individ. instances.
  void SetPose(const Eigen::Matrix4f &new_pose) {
    *(this->last_egomotion_) = ItmToEigen(this->trackingState->pose_d->GetInvM()).inverse() * new_pose;
    this->trackingState->pose_d->SetInvM(EigenToItm(new_pose));
  }

  void Integrate() {
    this->denseMapper->ProcessFrame(
      // We already generate our new view when splitting the input based on the segmentation.
      // The tracking state is kept up-to-date by the tracker.
      // The scene actually holds the voxel hash. It's almost a vanilla struct.
      // The render state is used for things like raycasting
      this->view, this->trackingState, this->scene, this->renderState_live);
  }

  void PrepareNextStep() {
    // This may not be necessary if we're using ground truth VO.
    this->trackingController->Prepare(this->trackingState, this->view, this->renderState_live);

    // Keep the OpenCV previews up to date.
    ItmToCv(*this->view->rgb, rgb_cv_);
    ItmToCv(*this->view->depth, raw_depth_cv_);
  }

  const ITMLibSettings* GetSettings() const {
    return settings;
  }

  // Not const because 'ITMMainEngine's implementation is not const either.
  void GetImage(
      ITMUChar4Image *out,
      dynslam::PreviewType get_image_type,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix()
  );

  /// \brief Returns the RGB "seen" by this particular InfiniTAM instance.
  /// This may not be the full original RGB frame due to, e.g., masking.
  const cv::Mat3b* GetRgbPreview() const {
    return rgb_cv_;
  }

  /// \brief Returns the depth "seen" by this particular InfiniTAM instance.
  /// This may not be the full original depth frame due to, e.g., masking.
  const cv::Mat1s* GetDepthPreview() const {
    return raw_depth_cv_;
  }

  Eigen::Matrix4f GetPose() const {
    return ItmToEigen(trackingState->pose_d->GetInvM());
  }

  /// \brief Returns the transform from the previous frame to the current.
  Eigen::Matrix4f GetLastEgomotion() const {
    return *last_egomotion_;
  }

  /// \brief Regularizes the map by pruning low-weight voxels which are old enough.
  /// Very useful for, e.g., reducing artifacts caused by noisy depth maps.
  /// \note This implementation is only supported on the GPU, and for voxel hashing.
  void Decay() {
    denseMapper->Decay(scene, max_decay_weight_, min_decay_age_, false);
  }

  /// \brief Aggressive decay which ignores the minimum age requirement and acts on ALL voxels.
  /// Typically used to clean up finished reconstructions. Can be much slower than `Decay`, even by
  /// a few orders of magnitude if used on the full static map.
  void Reap() {
    denseMapper->Decay(scene, max_decay_weight_, 0, true);
  }

  size_t GetVoxelSizeBytes() const {
    return sizeof(ITMVoxel);
  }

  size_t GetUsedMemoryBytes() const {
    int num_used_blocks = (scene->index.getNumAllocatedVoxelBlocks() - scene->localVBA.lastFreeBlockId);
    return GetVoxelSizeBytes() * SDF_BLOCK_SIZE3 * num_used_blocks;
  }

  size_t GetSavedDecayMemory() const {
    size_t block_size_bytes = GetVoxelSizeBytes() * SDF_BLOCK_SIZE3;
    size_t decayed_block_count = denseMapper->GetDecayedBlockCount();
    return decayed_block_count * block_size_bytes;
  }

  // Necessary for having Eigen types as fields.
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  ITMUChar4Image *rgb_itm_;
  ITMShortImage  *raw_depth_itm_;

  cv::Mat3b *rgb_cv_;
  cv::Mat1s *raw_depth_cv_;

  Eigen::Matrix4f *last_egomotion_;

  // Parameters for voxel decay
  // w=3, a=5 seems a little aggressive for dispnet. As long as we're using it and not elas, maybe
  // even w=1, a=7-8 can also work.
  // ELAS
//  int max_decay_weight_ = 3;
//  int min_decay_age_ = 10;

  // Semi-aggressive debug
    int max_decay_weight_= 2;
    int min_decay_age_ = 30;
  /// \brief Voxels older than this are eligible for decay.
//  int min_decay_age_ = 10;
  /// \brief Voxels with a weight smaller than this are decayed, provided that they are old enough.
//  int max_decay_weight_ = 2;
};

} // namespace drivers
} // namespace dynslam


#endif //DYNSLAM_INFINITAMDRIVER_H
