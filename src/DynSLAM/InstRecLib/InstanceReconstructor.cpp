
#include <algorithm>
#include <Eigen/StdVector>

#include "InstanceReconstructor.h"
#include "InstanceView.h"
#include "../DynSlam.h"
#include "../../libviso2/src/viso.h"
#include "../Direct/frame/device/cpu/frame_cpu.h"
#include "../Direct/frame/frame.hpp"
#include "../Direct/pinholeCameraModel.h"
#include "../Direct/image_alignment/device/cpu/dirImgAlignCPU.h"

namespace instreclib {
namespace reconstruction {

using namespace std;

using namespace dynslam::utils;
using namespace instreclib::segmentation;
using namespace instreclib::utils;

using namespace ITMLib::Objects;


const vector<string> InstanceReconstructor::kClassesToReconstructVoc2012 = { "car", "bus" };
// Note: for a real self-driving cars, you definitely want a completely generic obstacle detector.
const vector<string> InstanceReconstructor::kPossiblyDynamicClassesVoc2012 = {
    "airplane",   // you never know...
    "bicycle",
    "bird",       // stupid pigeons
    "boat",       // again, you never know...
    "bus",
    "car",
    "cat",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "sheep",      // If we're driving in Romania.
    "train"       // MNC is not very good at segmenting trains anyway
};

const vector<Eigen::Vector4i, Eigen::aligned_allocator<Eigen::Vector4i>> InstanceReconstructor::kMatplotlib2Palette = {
    Eigen::Vector4i(0x1f, 0x77, 0xb4, 255),
    Eigen::Vector4i(0xff, 0x7f, 0x0e, 255),
    Eigen::Vector4i(0x2c, 0xa0, 0x2c, 255),
    Eigen::Vector4i(0xd6, 0x27, 0x28, 255),
    Eigen::Vector4i(0x94, 0x67, 0xbd, 255),
    Eigen::Vector4i(0x8c, 0x56, 0x4b, 255),
    Eigen::Vector4i(0xe3, 0x77, 0xc2, 255),
    Eigen::Vector4i(0x71, 0x71, 0x71, 255),
    Eigen::Vector4i(0xbc, 0xbd, 0x22, 255),
    Eigen::Vector4i(0x17, 0xbe, 0xcf, 255)
};

// TODO(andrei): Implement this in CUDA. It should be easy.
// TODO(andrei): Consider renaming to copysilhouette or something.
template <typename DEPTH_T>
void ProcessSilhouette_CPU(Vector4u *source_rgb,
                           DEPTH_T *source_depth,
                           Vector4u *dest_rgb,
                           DEPTH_T *dest_depth,
                           Eigen::Vector2i sourceDims,
                           const Mask &copy_mask,
                           const Mask &delete_mask) {
  // Blanks out the detection's silhouette in the 'source' frames, and writes its pixels into the
  // output frames. Initially, the dest frames will be the same size as the source ones, but this
  // is wasteful in terms of memory: we should use bbox+1-sized buffers in the future, since most
  // silhouettes are relatively small wrt the size of the whole frame.
  //
  // Moreover, we should be able to pass in several output buffer addresses and a list of
  // detections to the CUDA kernel, and do all the ``splitting up'' work in one kernel call. We
  // may need to add support for the adaptive-size output buffers, since otherwise writing to
  // e.g., 5-6 full-size output buffers from the same kernel may end up using up way too much GPU
  // memory.

  // TODO(andrei): Check if this assumption is still necessary.
  // Assumes that the copy mask is larger than the delete mask.

  int frame_width = sourceDims[0];
  int frame_height = sourceDims[1];
  const BoundingBox &copy_bbox = copy_mask.GetBoundingBox();
//  const BoundingBox &delete_bbox = delete_mask.GetBoundingBox();

  int copy_box_width = copy_bbox.GetWidth();
  int copy_box_height = copy_bbox.GetHeight();
//  int delete_box_width = delete_bbox.GetWidth();
//  int delete_box_height = delete_bbox.GetHeight();

  memset(dest_rgb, 255, frame_width * frame_height * sizeof(*source_rgb));
  memset(dest_depth, 0, frame_width * frame_height * sizeof(DEPTH_T));

  // Keep track of the minimum depth in the frame, so we can use it as a heuristic when
  // reconstructing instances.
  float min_depth = numeric_limits<float>::max();
  const float kInvalidDepth = -1.0f;

  for (int row = 0; row < copy_box_height; ++row) {
    for (int col = 0; col < copy_box_width; ++col) {
      int copy_row = row + copy_bbox.r.y0;
      int copy_col = col + copy_bbox.r.x0;

      if (copy_row < 0 || copy_row >= frame_height || copy_col < 0 || copy_col >= frame_width) {
        continue;
      }

      int copy_idx = copy_row * frame_width + copy_col;
      u_char copy_mask_val = copy_mask.GetData()->at<u_char>(row, col);
      if (copy_mask_val == 1) {
        dest_rgb[copy_idx].r = source_rgb[copy_idx].r;
        dest_rgb[copy_idx].g = source_rgb[copy_idx].g;
        dest_rgb[copy_idx].b = source_rgb[copy_idx].b;
        dest_rgb[copy_idx].a = source_rgb[copy_idx].a;
        float depth = source_depth[copy_idx];
        dest_depth[copy_idx] = depth;

        if (depth != kInvalidDepth && depth < min_depth) {
          min_depth = depth;
        }
      }
      else {
        dest_rgb[copy_idx].r = 255;
        dest_rgb[copy_idx].g = 255;
        dest_rgb[copy_idx].b = 255;
      }
    }
  }


  // TODO(andrei): Store min depth somewhere.
//  cout << "Instance frame min depth: " << min_depth << endl;
}

template<typename TDepth>
void RemoveSilhouette_CPU(ORUtils::Vector4<unsigned char> *source_rgb,
                          TDepth *source_depth,
                          Eigen::Vector2i source_dimensions,
                          const Mask &mask) {

  int frame_width = source_dimensions[0];
  int frame_height = source_dimensions[1];
  const BoundingBox &bbox = mask.GetBoundingBox();

  int box_width = bbox.GetWidth();
  int box_height = bbox.GetHeight();

  for (int row = 0; row < box_height; ++row) {
    for (int col = 0; col < box_width; ++col) {
      int frame_row = row + bbox.r.y0;
      int frame_col = col + bbox.r.x0;

      if (frame_row < 0 || frame_row >= frame_height ||
          frame_col < 0 || frame_col >= frame_width) {
        continue;
      }

      int frame_idx = frame_row * frame_width + frame_col;
      u_char mask_val = mask.GetData()->at<u_char>(row, col);
      if (mask_val == 1) {
        source_rgb[frame_idx].r = 0;
        source_rgb[frame_idx].g = 0;
        source_rgb[frame_idx].b = 0;
        source_rgb[frame_idx].a = 0;
        source_depth[frame_idx] = 0.0f;
      }
    }
  }

}

void InstanceReconstructor::ProcessFrame(
    const dynslam::DynSlam *dyn_slam,
    ITMLib::Objects::ITMView *main_view,
    const segmentation::InstanceSegmentationResult &segmentation_result,
    const SparseSceneFlow &scene_flow,
    const SparseSFProvider &ssf_provider,
    bool always_separate
) {
  main_view->rgb->UpdateHostFromDevice();
  main_view->depth->UpdateHostFromDevice();

  Vector2i frame_size_itm = main_view->rgb->noDims;
  Eigen::Vector2i frame_size(frame_size_itm.x, frame_size_itm.y);
  vector<InstanceView, Eigen::aligned_allocator<InstanceView>> new_instance_views =
      CreateInstanceViews(segmentation_result, main_view, scene_flow);

  // Associate this frame's detection(s) with those from previous frames.
  this->instance_tracker_->ProcessInstanceViews(frame_idx_, new_instance_views, dyn_slam->GetPose());
  // Estimate relative object motion and update track states.
  this->UpdateTracks(dyn_slam, scene_flow, ssf_provider, always_separate, main_view, frame_size);
  // Update 3D models for tracks undergoing reconstruction.
  this->ProcessReconstructions(always_separate);

  // Update the GPU image after we've (if applicable) removed the dynamic objects from it.
  main_view->rgb->UpdateDeviceFromHost();
  main_view->depth->UpdateDeviceFromHost();

  /*
  // ``Graphically'' display the object tracks for debugging.
  for (const auto &pair: this->instance_tracker_->GetActiveTracks()) {
    cout << "Track: " << pair.second.GetAsciiArt() << endl;
  }
  // */

  frame_idx_++;
}

void InstanceReconstructor::UpdateTracks(const dynslam::DynSlam *dyn_slam,
                                         const SparseSceneFlow &scene_flow,
                                         const SparseSFProvider &ssf_provider,
                                         bool always_separate,
                                         ITMLib::Objects::ITMView *main_view,
                                         const Eigen::Vector2i &frame_size) const
{
  for (const auto &pair : instance_tracker_->GetActiveTracks()) {
    Track &track = instance_tracker_->GetTrack(pair.first);
    bool verbose = true;
    track.Update(dyn_slam->GetLastEgomotion(), ssf_provider, verbose);
    if (track.GetLastFrame().frame_idx == dyn_slam->GetCurrentFrameNo() - 1) {
      ProcessSilhouette(track, main_view, frame_size, scene_flow, always_separate);
    }
  }
}

void InstanceReconstructor::ProcessSilhouette(Track &track,
                                              ITMLib::Objects::ITMView *main_view,
                                              const Eigen::Vector2i &frame_size,
                                              const SparseSceneFlow &scene_flow,
                                              bool always_separate) const
{
  bool should_reconstruct = ShouldReconstruct(track.GetClassName());
  bool possibly_dynamic = IsPossiblyDynamic(track.GetClassName());
  auto &latest_frame = track.GetLastFrame();
  auto *instance_view = latest_frame.instance_view.GetView();
  assert(instance_view != nullptr);

  ORUtils::Vector4<uchar> *rgb_data_h = main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

  InstanceDetection &latest_detection = latest_frame.instance_view.GetInstanceDetection();
  if (track.GetState() == kUncertain) {
      if (possibly_dynamic) {
        printf("Unknown motion for possibly dynamic object of class %s; cutting away!\n",
               track.GetClassName().c_str());
        RemoveSilhouette_CPU(rgb_data_h, depth_data_h, frame_size,
                             *(latest_detection.delete_mask));
      }
      // else: Static class with unknown motion. Likely safe to put in main map.
    }
    else if (track.GetState() == kDynamic || always_separate) {
      if (should_reconstruct) {
        // Dynamic object which we should reconstruct, such as a car.
        auto rgb_segment_h = instance_view->rgb->GetData(MEMORYDEVICE_CPU);
        auto depth_segment_h = instance_view->depth->GetData(MEMORYDEVICE_CPU);
        ProcessSilhouette_CPU(rgb_data_h, depth_data_h,
                              rgb_segment_h, depth_segment_h,
                              frame_size,
                              *(latest_detection.copy_mask), *(latest_detection.delete_mask));
        // TODO(andrei): Really think whether there's a clean way of doing these two operations together.
        RemoveSilhouette_CPU(rgb_data_h, depth_data_h, frame_size, *(latest_detection.delete_mask));
        instance_view->rgb->UpdateDeviceFromHost();
        instance_view->depth->UpdateDeviceFromHost();
      }
      else if (possibly_dynamic) {
        cout << "Dynamic object with known motion we can't reconstruct. Removing." << endl;
        // Dynamic object which we can't or don't want to reconstruct, such as a pedestrian.
        // In this case, we simply remove the object from view.
        RemoveSilhouette_CPU(rgb_data_h, depth_data_h, frame_size,
                             *(latest_detection.delete_mask));
      }
      else {
        // Warn if we detect a moving potted plant.
        cerr << "Warning: found dynamic object of class [" << track.GetClassName() << "], which is "
             << "an unexpected type of dynamic object." << endl;
      }
    }
    else if (track.GetState() == kStatic) {
      // The object is known to be static; we don't have to do anything.
      return;
    }
    else {
      throw runtime_error("Unexpected track state.");
    }
}

ITMUChar4Image *InstanceReconstructor::GetInstancePreviewRGB(size_t track_idx) {
  if (! instance_tracker_->HasTrack(track_idx)) {
    return nullptr;
  }

  auto *view = instance_tracker_->GetTrack(track_idx).GetLastFrame().instance_view.GetView();
  if (view == nullptr) {
    return nullptr;
  }
  else {
    return view->rgb;
  }
}

ITMFloatImage *InstanceReconstructor::GetInstancePreviewDepth(size_t track_idx) {
  if (! instance_tracker_->HasTrack(track_idx)) {
    return nullptr;
  }

  auto *view = instance_tracker_->GetTrack(track_idx).GetLastFrame().instance_view.GetView();
  if (view == nullptr) {
    return nullptr;
  }
  else {
    return view->depth;
  }
}

void InstanceReconstructor::ProcessReconstructions(bool always_separate) {
  for (const auto &pair : instance_tracker_->GetActiveTracks()) {
    Track& track = instance_tracker_->GetTrack(pair.first);
    if (! ShouldReconstruct(track.GetClassName())) {
      continue;
    }

    // If we don't have any new information in this track, there's nothing to fuse.
    if(track.GetLastFrame().frame_idx != frame_idx_) {
      // TODO(andrei): Do this in a smarter way.
      // However, if we just encountered a gap in the track (or its end), let's do a full,
      // aggressive cleanup of the reconstruction, if applicable.
      int gap_size = frame_idx_ - track.GetLastFrame().frame_idx;
      if (track.NeedsCleanup() && track.HasReconstruction() && gap_size >= 2) {
        Tic(Format("Full cleanup for instance %d last seen at frame %d, so %d frame(s) ago.",
                   track.GetId(),
                   track.GetLastFrame().frame_idx,
                   gap_size));

        track.ReapReconstruction();
        TocMicro();

        track.SetNeedsCleanup(false);
      }

      continue;
    }

    if (! track.HasReconstruction()) {
      bool eligible = track.EligibleForReconstruction() &&
          (track.GetState() == TrackState::kDynamic || (track.GetState() == TrackState::kStatic && always_separate));

      if (eligible) {
        // No reconstruction allocated yet; let's initialize one.
        InitializeReconstruction(track);
      }
      else {
        // The frame data we have is insufficient, so we won't try to reconstruct the object
        // (yet).
        continue;
      }
    } else {
      // Fuse the latest frame into the volume.
      FuseFrame(track, track.GetSize() - 1);
    }
  }
}

void InstanceReconstructor::InitializeReconstruction(Track &track) const {
  cout << endl << "Starting to reconstruct instance with ID: " << track.GetId() << endl << endl;
  ITMLibSettings *settings = new ITMLibSettings(*driver_->GetSettings());

  // Set a much smaller voxel block number for the reconstruction, since individual objects
  // occupy a limited amount of space in the scene.
  // We don't want to create an (expensive) meshing engine for every instance.
  settings->createMeshingEngine = false;

  settings->sceneParams.mu = 1.00f;
  settings->sceneParams.voxelSize = 0.035f;
//  settings->sceneParams.voxelSize = 0.250f;

  // Can fix spurious hole issues (rare)
//  settings->sceneParams.voxelSize = 0.030f;
  // Volume approximated in meters.
  settings->sdfLocalBlockNum = static_cast<long>(5 * 5 * 10 / settings->sceneParams.voxelSize);
//  settings->sdfLocalBlockNum = static_cast<long>(4 * 4 * 10 / settings->sceneParams.voxelSize);

  track.GetReconstruction() = make_shared<InfiniTamDriver>(
          settings,
          driver_->GetView()->calib,
          driver_->GetView()->rgb->noDims,
          driver_->GetView()->rgb->noDims,
          driver_->GetVoxelDecayParams(),
          driver_->IsUsingDepthWeights()
  );

  // TODO(andrei): This may not work for the (Stat/dyn) -> Unc -> (stat/dyn) situation!
  // If we already have some frames, integrate them into the new volume.
  int first_idx = track.GetFirstFusableFrameIndex();
  if (first_idx > -1) {
    cout << "Starting reconstruction from index " << first_idx << endl;
    cout << "Camera pose for that index:" << endl << track.GetFrame(first_idx).camera_pose << endl << endl;
    for (size_t i = first_idx; i < track.GetSize(); ++i) {
      FuseFrame(track, i);
    }
  }
}

/// \brief Converts an 8-bit RGB color to an 8-bit grayscale intensity.
uchar RgbToGrayscale(uchar r, uchar g, uchar b) {
  return static_cast<uchar>(r * 0.299 + g * 0.587 + b * 0.114);
}

/// \brief Converts a given (intensity, depth) frame into a list of "depth hypotheses" to be used
///        in the direct alignment code.
/// This function basically just massages data from the DynSLAM side into the format required in the
/// direct alignment code, which is based on Liu et al., 2017 "Direct Visual Odometry for a
/// Fisheye-Stereo Camera."
vector<DepthHypothesis_GMM> GenHyps(const uchar *intensity, const float *depth, CameraBase &camera,
                                    int rows, int cols) {
  // Note: variance not used in depth alignment code, so there's no point in trying to estimate it
  // from, e.g., the depth.
  vector<DepthHypothesis_GMM> hypotheses;

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;

      if (depth[idx] < 0.01f) {
        continue;
      }

      DepthHypothesis_GMM hypothesis;
      hypothesis.pixel[0] = row;
      hypothesis.pixel[1] = col;
      camera.backProject(row, col, hypothesis.unitRay);
      hypothesis.intensity = intensity[idx];
      hypothesis.rayDepth = depth[idx];
      hypothesis.bValidated = true;

      hypotheses.push_back(hypothesis);
    }
  }

  return hypotheses;
}

uchar* RgbToGrayscaleImage(const Vector4u *rgb_image, int rows, int cols) {
  auto grayscale_image = new uchar[rows * cols];

  for(int row = 0; row < rows; ++row) {
    for(int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      grayscale_image[idx] = RgbToGrayscale(rgb_image[idx].r, rgb_image[idx].g, rgb_image[idx].b);
    }
  }

  return grayscale_image;
}

/// \brief Refines an existing pose using a direct method.
/// The actual pose we're starting from is hidden in the latest frame of the track.
/// \return Whether the refinement could run.
bool ExperimentalDirectRefine(Track &track,
                              int first_idx,
                              int second_idx,
                              Eigen::Matrix4d &relative_pose,   // TODO do we still need this?
                              const ITMRGBDCalib &calib,
                              Eigen::Matrix4f &out_refined_pose) {
  cout << "Will run experimental direct alignment pose refinement. " << endl;
  cout << "Initial estimate (from sparse RANSAC): " << endl << relative_pose << endl;
  auto first_view = track.GetFrame(first_idx).instance_view.GetView();
  auto second_view = track.GetFrame(second_idx).instance_view.GetView();
  auto first_depth = first_view->depth;
  auto second_depth = second_view->depth;
  auto first_rgb = first_view->rgb;
  auto second_rgb = second_view->rgb;
  first_depth->UpdateHostFromDevice();
  second_depth->UpdateHostFromDevice();
  first_rgb->UpdateHostFromDevice();
  second_rgb->UpdateHostFromDevice();

  float *ff = first_depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  float *sf = second_depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  Vector4u *f_rgb = first_rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  Vector4u *s_rgb = second_rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

  using namespace VGUGV::Common;
  using namespace VGUGV::SLAM;

  int rows = first_depth->noDims.y;
  int cols = first_depth->noDims.x;
  int channels = 1;
  Eigen::Vector2i frame_size(rows, cols);
  Eigen::Matrix3f K;
  auto &params = calib.intrinsics_rgb.projectionParamsSimple;
  K << params.fx, 0,          params.px,
       0,         params.fy,  params.py,
       0,                 0,          1;
  CameraBase::Ptr cam = make_shared<PinholeCameraModel>(frame_size, K);
  cout << "Performing direct alignment for frames " << first_idx << " and " << second_idx
       << " from track #" << track.GetId() << ". Frame size: " << cols << " x " << rows
       << ". K: " << endl << K << endl;

  uchar *uchar_data_first = RgbToGrayscaleImage(f_rgb, rows, cols);
  uchar *uchar_mask_first = nullptr;
  uchar *uchar_data_second = RgbToGrayscaleImage(s_rgb, rows, cols);
  uchar *uchar_mask_second = nullptr;

  cout << "Generating hypotheses..." << endl;
  vector<DepthHypothesis_GMM> hyps_first = GenHyps(uchar_data_first, ff, *cam, rows, cols);
  vector<DepthHypothesis_GMM> hyps_second = GenHyps(uchar_data_second, sf, *cam, rows, cols);

  if (hyps_first.size() < 10 || hyps_second.size() < 10) {
    cout << "Hyps too small: first.size() == " << hyps_first.size() << "; second.size() == "
         << hyps_second.size() << endl;
    return false;
  }
  else {
    cout << "Enough hyps: first.size() == " << hyps_first.size() << "; second.size() == "
         << hyps_second.size() << " Will optimize!" << endl;
  }

  auto first_ddm = make_shared<FrameCPU_denseDepthMap>(
      first_idx, cam, uchar_data_first, uchar_mask_first, rows, cols, channels);
  auto second_ddm = make_shared<FrameCPU_denseDepthMap>(
      second_idx, cam, uchar_data_second, uchar_mask_second, rows, cols, channels);

  // XXX: the top pyramid level is distorted!! when n = 4 or 3. For 2 and 1 it looks OK.
  int nMaxPyramidLevels = 2;
  first_ddm->computeImagePyramids(nMaxPyramidLevels);
  second_ddm->computeImagePyramids(nMaxPyramidLevels);
  first_ddm->computeImagePyramidsGradients(nMaxPyramidLevels);
  second_ddm->computeImagePyramidsGradients(nMaxPyramidLevels);

  first_ddm->copyFeatureDescriptors(hyps_first.data(), hyps_first.size(), nMaxPyramidLevels);
  second_ddm->copyFeatureDescriptors(hyps_second.data(), hyps_second.size(), nMaxPyramidLevels);

  // TODO(andrei): If refinement fails because, e.g., the top of the pyramid is already too tiny,
  // then simply drop the process, and only use the RANSAC estimate, or disable fusion completely.
  // smaller -> penalize outlier harder
  float huber_delta = 1.0f;
  int nMaxIterations = 50;
  float epsilon = 1e-7;
  float robust_loss_param = huber_delta;
  float min_gradient_magnitude = 2.0f;

  DirImgAlignCPU dir_img_align(nMaxPyramidLevels, nMaxIterations, epsilon,
                               ROBUST_LOSS_TYPE::PSEUDO_HUBER, robust_loss_param,
                               min_gradient_magnitude);

  Transformation transformation;
  transformation.setT(track.GetFrame(second_idx).relative_pose->Get().matrix_form.cast<float>());
  cout << "Will pass the following relative pose transformation to the direct part: " << endl
      << transformation.getTMatrix() << endl;

  dir_img_align.doAlignment(first_ddm, second_ddm, transformation);

  cout << "Direct alignment done. Old was: " << endl
       << track.GetFrame(second_idx).relative_pose->Get().matrix_form.cast<float>() << endl;
  cout << "Transformation after direct alignment:" << endl << transformation.getTMatrix() << endl;

  out_refined_pose = transformation.getTMatrix();

  delete uchar_data_first;
  delete uchar_data_second;
  return true;
}


void InstanceReconstructor::FuseFrame(Track &track, size_t frame_idx) const {
  if (track.GetState() == TrackState::kUncertain) {
    // We can't deal with tracks of uncertain state, because there's no available relative
    // transforms between frames, so we can't register measurements.
    return;
  }

  cout << "Processing reconstruction of instance with ID: " << track.GetId() << endl;
  InfiniTamDriver &instance_driver = *track.GetReconstruction();

  TrackFrame &frame = track.GetFrame(frame_idx);
  instance_driver.SetView(frame.instance_view.GetView());
  Option<Eigen::Matrix4d> rel_dyn_pose = track.GetFramePose(frame_idx);

  // Only fuse the information if the relative pose could be established.
  // TODO-LOW(andrei): This would require modifying libviso a little, but in the event that we miss
  // a frame, i.e., we are unable to estimate the relative pose from frame k to frame k+1, we could
  // still try to estimate it from k to k+2.
  if (rel_dyn_pose.IsPresent()) {
    Eigen::Matrix4f rel_dyn_pose_f = (*rel_dyn_pose).cast<float>();
    cout << "Fusing frame " << frame_idx << "/ #" << track.GetId() << "." << endl << rel_dyn_pose_f << endl;
    instance_driver.SetPose(rel_dyn_pose_f.inverse());

    if (enable_direct_refinement_ && enable_itm_refinement_) {
      throw std::runtime_error("Cannot use both direct image alignment AND the ITM-specific tracker(s)!");
    }

    if (enable_direct_refinement_ && frame_idx > 0 && frame.relative_pose->IsPresent()) {
      vector<double> unrefined_se3 = frame.relative_pose->Get().se3_form;

      // Ensure we have a previous frame to align to, and do the direct alignment.
      Eigen::Matrix4f new_relative_pose_matrix;
      // TODO If that doesn't work, try to align against a raycast of the model...
      bool success = ExperimentalDirectRefine(track,
                               static_cast<int>(frame_idx - 1),
                               static_cast<int>(frame_idx),
                               rel_dyn_pose.Get(),
                               *(frame.instance_view.GetView()->calib),
                               new_relative_pose_matrix);
      if(success) {
        delete frame.relative_pose;
        // TODO same as before... try updating se3 representation.
        // TODO be more consistent with float/double
        frame.relative_pose = new Option<Pose>(new Pose(
            unrefined_se3,
            new_relative_pose_matrix.cast<double>()
        ));

        rel_dyn_pose_f = track.GetFramePose(frame_idx).Get().cast<float>();   // TODO(andrei): Make this less slow.
        instance_driver.SetPose(rel_dyn_pose_f.inverse());
      }
    }

    if (enable_itm_refinement_) {
      vector<double> unrefined_se3 = frame.relative_pose->Get().se3_form;

      // This should, in theory, try to refine the pose even further...
      instance_driver.Track();

      if (frame.relative_pose->IsPresent()) {
        Eigen::Matrix4d old_rel_pose = frame.relative_pose->Get().matrix_form;

        Eigen::Matrix4d new_pose = instance_driver.GetPose().cast<double>().inverse();
        Eigen::Matrix4d delta = new_pose * rel_dyn_pose.Get();
        Eigen::Matrix4d refined_matrix = old_rel_pose * delta;

        cout << "Refined matrix inv: " << refined_matrix.inverse();

        // TODO(andrei): The improvement may not be significant, but we should also update the se3 form
        delete frame.relative_pose;
        frame.relative_pose = new Option<Pose>(new Pose(
            unrefined_se3,
            refined_matrix
//          old_rel_pose
        ));

        cout << "Frame " << frame_idx << ": Refined relative pose only. " << endl
             << "Old relative: " << endl
             << old_rel_pose << endl << "New relative, refined by ICP: " << endl
             << refined_matrix << endl;
        cout << "Sanity checks:" << endl << frame.relative_pose->Get().matrix_form << endl;
        for (int i = 0; i < 6; ++i) {
          cout << frame.relative_pose->Get().se3_form[i] << ", ";
        }
        cout << endl;
      } else {
        cout << "Frame " << frame_idx << " had no relative pose, so it could not be refined."
             << endl;
      }
    }

    try {
      instance_driver.Integrate();
    }
    catch (runtime_error &error) {
      // TODO(andrei): Custom dynslam allocation exception we can catch here to avoid fatal errors.
      // This happens when we run out of memory on the GPU for this volume. We should prolly have a
      // custom exception/error code for this.
      cerr << "Caught runtime error while integrating new data into an instance volume: "
           << error.what() << endl << "Will continue regular operation." << endl;
    }

    instance_driver.PrepareNextStep();

    // TODO(andrei): Make this sync with the similar flag in 'DynSlam'.
    if (use_decay_) {
      instance_driver.Decay();
    }

    track.SetNeedsCleanup(true);
    track.CountFusedFrame();

    // Note: need to discard older ones, since we need cur+prev for direct alignment.
    // Free up memory now that we've fused the frame!
//    frame.instance_view.DiscardView();

    // TODO remove if unnecessary (cleanup); NOTE: may be useful since we may prefer to keep the
    // most recent view even after the fusion, for visualization purposes.
//    // Free up memory from the previous frame.
    int prev_idx = static_cast<int>(frame_idx) - 1;
    if (prev_idx >= 0) {
      auto prev_frame = track.GetFrame(prev_idx);
      prev_frame.instance_view.DiscardView();
    }
  }
  else {
    cout << "Could not fuse instance data for track #" << track.GetId() << " due to missing pose "
         << "information." << endl;
  }
}

void InstanceReconstructor::GetInstanceRaycastPreview(ITMUChar4Image *out,
                                                      int object_idx,
                                                      const pangolin::OpenGlMatrix &model_view,
                                                      dynslam::PreviewType preview_type) {
  if (instance_tracker_->HasTrack(object_idx)) {
    Track& track = instance_tracker_->GetTrack(object_idx);
    if (track.HasReconstruction()) {
      track.GetReconstruction()->GetImage(
          out,
          preview_type,
          model_view);

      return;
    }
  }

  // If we're not tracking the object, or if it has no reconstruction available, then there's
  // nothing to display.
  out->Clear();
}

void InstanceReconstructor::ForceObjectCleanup(int object_id) {
  if(! instance_tracker_->HasTrack(object_id)) {
    throw std::runtime_error(dynslam::utils::Format("Unknown track ID: %d", object_id));
  }

  Track& track = instance_tracker_->GetTrack(object_id);
  if(! track.HasReconstruction()) {
    throw std::runtime_error("Track exists but has no reconstruction.");
  }

  track.ReapReconstruction();
}

void InstanceReconstructor::SaveObjectToMesh(int object_id, const string &fpath) {
  if(! instance_tracker_->HasTrack(object_id)) {
    throw std::runtime_error(dynslam::utils::Format("Unknown track ID: %d", object_id));
  }

  const Track& track = instance_tracker_->GetTrack(object_id);
  if(! track.HasReconstruction()) {
    throw std::runtime_error("Track exists but has no reconstruction.");
  }

  // TODO(andrei): Wrap this meshing code inside a nice utility.
  // Begin ITM-specific meshing code
  const ITMLibSettings *settings = track.GetReconstruction()->GetSettings();
  auto *meshing_engine = new ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>(
      settings->sdfLocalBlockNum);
  track.GetReconstruction()->GetScene();

  MemoryDeviceType deviceType = (settings->deviceType == ITMLibSettings::DEVICE_CUDA
                                 ? MEMORYDEVICE_CUDA
                                 : MEMORYDEVICE_CPU);
  ITMMesh *mesh = new ITMMesh(deviceType, settings->sdfLocalBlockNum);
  meshing_engine->MeshScene(mesh, track.GetReconstruction()->GetScene());
  mesh->WriteOBJ(fpath.c_str());
//    mesh->WriteSTL(fpath.c_str());

  delete mesh;
  delete meshing_engine;
}

vector<InstanceView, Eigen::aligned_allocator<InstanceView>> InstanceReconstructor::CreateInstanceViews(
    const InstanceSegmentationResult &segmentation_result,
    ITMLib::Objects::ITMView *main_view,
    const SparseSceneFlow &scene_flow
) {
  Vector2i frame_size_itm = main_view->rgb->noDims;
  Eigen::Vector2i frame_size(frame_size_itm.x, frame_size_itm.y);

  vector<InstanceView, Eigen::aligned_allocator<InstanceView>> instance_views;
  for (const InstanceDetection &instance_detection : segmentation_result.instance_detections) {
    if (IsPossiblyDynamic(instance_detection.GetClassName())) {
      // This is probably better; TODO(andrei): Dig into this after deadline.
//    if (ShouldReconstruct(instance_detection.GetClassName())) {
      // bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
      // check this, since the field is private.
      bool use_gpu = true;

      // The ITMView takes ownership of this.
      ITMRGBDCalib *calibration = new ITMRGBDCalib;
      *calibration = *main_view->calib;
      auto view = make_shared<ITMView>(calibration, frame_size_itm, frame_size_itm, use_gpu);
      vector<RawFlow, Eigen::aligned_allocator<RawFlow>> instance_flow_vectors;
      ExtractSceneFlow(
          scene_flow,
          instance_flow_vectors,
          instance_detection,
          frame_size);

      instance_views.emplace_back(instance_detection,
                                  view,
                                  instance_flow_vectors);
    }
  }

  return instance_views;
}

void InstanceReconstructor::ExtractSceneFlow(const SparseSceneFlow &scene_flow,
                                             vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &out_instance_flow_vectors,
                                             const InstanceDetection &detection,
                                             const Eigen::Vector2i &frame_size,
                                             bool check_sf_start) {
  auto flow_mask = detection.delete_mask;
  const BoundingBox &flow_bbox = flow_mask->GetBoundingBox();
  map<pair<int, int>, RawFlow> coord_to_flow;
  int frame_width = frame_size(0);
  int frame_height = frame_size(1);

  for(const auto &match : scene_flow.matches) {
    int fx = static_cast<int>(match.curr_left(0));
    int fy = static_cast<int>(match.curr_left(1));
    int fx_prev = static_cast<int>(match.prev_left(0));
    int fy_prev = static_cast<int>(match.prev_left(1));

    if (flow_mask->ContainsPoint(fx, fy)) {
      // If checking the SF start, also ensure that keypoint position in the previous frame is also
      // inside the current mask. This limits the number of valid SF points, but also gets rid of
      // many noisy masks.
      if (!check_sf_start || detection.copy_mask->GetBoundingBox().ContainsPoint(fx_prev, fy_prev)) {
        coord_to_flow.emplace(pair<pair<int, int>, RawFlow>(pair<int, int>(fx, fy), match));
      }
    }
  }

  // TODO(andrei): This second half is NOT necessary and also slow... It should be removed
  // post-deadline.
  for (int cons_row = 0; cons_row < flow_bbox.GetHeight(); ++cons_row) {
    for (int cons_col = 0; cons_col < flow_bbox.GetWidth(); ++cons_col) {
      int cons_frame_row = cons_row + flow_bbox.r.y0;
      int const_frame_col = cons_col + flow_bbox.r.x0;

      if (cons_frame_row >= frame_height || const_frame_col >= frame_width) {
        continue;
      }

      u_char mask_val = flow_mask->GetData()->at<u_char>(cons_row, cons_col);
      if (mask_val == 1) {
        auto coord_pair = pair<int, int>(const_frame_col, cons_frame_row);
        if (coord_to_flow.find(coord_pair) != coord_to_flow.cend()) {
          out_instance_flow_vectors.push_back(coord_to_flow.find(coord_pair)->second);
        }
      }
    }
  }
}

void CompositeDepth(ITMFloatImage *target, const ITMFloatImage *source) {
  assert(target->noDims == source->noDims);

  float* t_data = target->GetData(MEMORYDEVICE_CPU);
  const float *s_data = source->GetData(MEMORYDEVICE_CPU);

  for(int i = 0; i < target->noDims[0]; i++) {
    for(int j = 0; j < target->noDims[1]; j++) {
      int idx = i * target->noDims[1] + j;

      if (t_data[idx] == 0) {
        t_data[idx] = s_data[idx];
      }
      else {
        if (s_data[idx] != 0) {
          t_data[idx] = min(t_data[idx], s_data[idx]);
        }
      }
    }
  }
}

/// \brief Adds an object instance to the reconstruction previeww.
/// Z-buffering in software as a first prototype (yes, very slow and silly).
void CompositeColor(ITMUChar4Image *target_color, ITMFloatImage *target_depth,
                    const ITMUChar4Image *instance_color, const ITMFloatImage *instance_depth,
                    const Eigen::Vector4i &tint, const float tint_strength
) {
  assert(target_color->noDims == target_depth->noDims &&
         target_color->noDims == instance_color->noDims &&
         instance_color->noDims == instance_depth->noDims);

  int width = target_color->noDims.width;
  int height = target_color->noDims.height;

  float* t_depth_data = target_depth->GetData(MEMORYDEVICE_CPU);
  const float *s_depth_data = instance_depth->GetData(MEMORYDEVICE_CPU);
  Vector4u *t_color_data = target_color->GetData(MEMORYDEVICE_CPU);
  const Vector4u *s_color_data = instance_color->GetData(MEMORYDEVICE_CPU);
  const float kColorBoost = 0.50f;

  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      int idx = i * width + j;

      bool instance_on_top = (s_depth_data[idx] != 0 &&
          (t_depth_data[idx] == 0 || t_depth_data[idx] > s_depth_data[idx]));

      if (instance_on_top) {
        t_depth_data[idx] = s_depth_data[idx];
        double col_strength = 1.0 + kColorBoost - tint_strength;
        t_color_data[idx].r = static_cast<uchar>(min(255.0, s_color_data[idx].r * col_strength + tint(0) * tint_strength));
        t_color_data[idx].g = static_cast<uchar>(min(255.0, s_color_data[idx].g * col_strength + tint(1) * tint_strength));
        t_color_data[idx].b = static_cast<uchar>(min(255.0, s_color_data[idx].b * col_strength + tint(2) * tint_strength));
      }
    }
  }
}


void InstanceReconstructor::CompositeInstanceDepthMaps(ITMFloatImage *out,
                                                       const pangolin::OpenGlMatrix &model_view) {
  // TODO(andrei): With a little bit of massaging, this could be implemented using the OpenGL zbuffer.
  // => much better perf.
  int current_frame_idx = this->frame_idx_;
  for(auto &entry : instance_tracker_->GetActiveTracks()) {
    Track &t = instance_tracker_->GetTrack(entry.first);

    if (t.GetLastFrame().frame_idx == current_frame_idx - 1 && t.HasReconstruction()) {
      Option<Eigen::Matrix4d> pose = t.GetFramePoseDeprecated(t.GetSize() - 1);
      if (pose.IsPresent()) {
        /// XXX: experimental freeview fused code. This works but may wreck the evaluation a little. Care is needed.
        auto pango_object_pose = model_view * pangolin::OpenGlMatrix::ColMajor4x4(pose.Get().data());
        t.GetReconstruction()->GetFloatImage(&instance_depth_buffer_,
                                             dynslam::PreviewType::kDepth,
                                             pango_object_pose);
        CompositeDepth(out, &instance_depth_buffer_);
      }
    }
  }
}

void InstanceReconstructor::CompositeInstances(ITMUChar4Image *out_color,
                                               ITMFloatImage *out_depth,
                                               dynslam::PreviewType preview_type,
                                               const pangolin::OpenGlMatrix &model_view) {
  int current_frame_idx = this->frame_idx_;
  const float kTintStrength = 1.00f;

  // TODO-LOW(andrei): consider compositing the most recent depth/color frames in the final view,
  // even if tracking in 3D is not yet successful. Should be OK for visualization, since the object
  // is segmented out and prevented from corrupting the actual map anyway.

  // Dim the background a little to highlight the instances better.
  float dim_factor = 0.10f;
  Vector4u *color_vals = out_color->GetData(MEMORYDEVICE_CPU);
  for (int i = 0; i < out_color->noDims.height; ++i) {
    for (int j = 0; j < out_color->noDims.width; ++j) {
      int idx = i * out_color->noDims.width + j;
      color_vals[idx].r = static_cast<uchar>(color_vals[idx].r * (1.0 - dim_factor));
      color_vals[idx].g = static_cast<uchar>(color_vals[idx].g * (1.0 - dim_factor));
      color_vals[idx].b = static_cast<uchar>(color_vals[idx].b * (1.0 - dim_factor));
    }
  }

  for (auto &entry : instance_tracker_->GetActiveTracks()) {
    Track &track = instance_tracker_->GetTrack(entry.first);

    // If an object is dynamic, we need precise pose info, so we can only show them if we've seen
    // them in the latest frame. Static objects don't impose this constraint.
    bool can_render_correctly = (track.GetLastFrame().frame_idx == current_frame_idx - 1 ||
        track.GetState() == TrackState::kStatic);

    if (can_render_correctly && track.HasReconstruction()) {
      auto pose = track.GetFramePoseDeprecated(track.GetSize() - 1);

      if (pose.IsPresent()) {
        auto pangolin_pose = model_view * pangolin::OpenGlMatrix::ColMajor4x4(pose.Get().data());
        track.GetReconstruction()->GetImage(
            &instance_color_buffer_,
            preview_type,
            pangolin_pose);
        track.GetReconstruction()->GetFloatImage(&instance_depth_buffer_,
                                                 dynslam::PreviewType::kDepth,
                                                 pangolin_pose);

        const Eigen::Vector4i &tint = kMatplotlib2Palette[track.GetId() % kMatplotlib2Palette.size()];
        CompositeColor(out_color,
                       out_depth,
                       &instance_color_buffer_,
                       &instance_depth_buffer_,
                       tint,
                       kTintStrength);
      }
    }

  }


}

}  // namespace reconstruction
}  // namespace instreclib
