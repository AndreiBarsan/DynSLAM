
#include <algorithm>
#include <vector>

#include "InstanceReconstructor.h"
#include "InstanceView.h"
#include "../DynSlam.h"
#include "../../libviso2/src/viso.h"

namespace instreclib {
namespace reconstruction {

using namespace std;

using namespace dynslam::utils;
using namespace instreclib::segmentation;
using namespace instreclib::utils;

using namespace ITMLib::Objects;


const vector<string> InstanceReconstructor::kClassesToReconstructVoc2012 = { "car" };
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

  memset(dest_rgb, 0, frame_width * frame_height * sizeof(*source_rgb));
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
  vector<InstanceView> new_instance_views = CreateInstanceViews(segmentation_result, main_view, scene_flow);

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
  const auto &tracks = instance_tracker_->GetActiveTracks();
  if (tracks.empty()) {
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
          (track.GetState() == TrackState::kDynamic || always_separate);

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
  cout << endl << endl;
  cout << "Starting to reconstruct instance with ID: " << track.GetId() << endl << endl;
  ITMLibSettings *settings = new ITMLibSettings(*driver_->GetSettings());

  // Set a much smaller voxel block number for the reconstruction, since individual objects
  // occupy a limited amount of space in the scene.
  // TODO(andrei): Set this limit based on some physical specification, such as 10m x 10m x
  // 10m.
  settings->sdfLocalBlockNum = 20000;
  // We don't want to create an (expensive) meshing engine for every instance.
  settings->createMeshingEngine = false;

  track.GetReconstruction() = make_shared<InfiniTamDriver>(
          settings,
          driver_->GetView()->calib,
          driver_->GetView()->rgb->noDims,
          driver_->GetView()->rgb->noDims);

  // TODO(andrei): This may not work for the (Stat/dyn) -> Unc -> (stat/dyn) situation!
  // If we already have some frames, integrate them into the new volume.
  int first_idx = track.GetFirstFusableFrameIndex();
  if (first_idx > -1) {
    cout << "Starting reconstruction from index " << first_idx << endl;
    for (size_t i = first_idx; i < track.GetSize(); ++i) {
      FuseFrame(track, i);
    }
  }
}

// TODO-LOW(andrei): IDEA: in poor man KF mode, of if using proper KF, can use the inverse
// (magnitude of?) the variance as an update weight. That is, if we, say, are unable to estimate
// relative motion for a particular vehicle over 1-2 frames, we can reduce the weight of subsequent
// updates.
// Similarly, we could even adjust it based on the final residual from the pose estimation/dense
// alignment.
void InstanceReconstructor::FuseFrame(Track &track, size_t frame_idx) const {
  if (track.GetState() == TrackState::kUncertain) {
    // We can't deal with tracks of uncertain state, because there's no available relative
    // transforms between frames, so we can't register measurements.
    return;
  }

  //  cout << "Continuing to reconstruct instance with ID: " << track.GetId() << endl;
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
//    cout << "Fusing frame " << frame_idx << "/ #" << track.GetId() << "." << endl << rel_dyn_pose_f << endl;

    // Note: We have the ITM instance up and running, so we can even perform ICP or some dense
    // alignment method here if we wish.
    instance_driver.SetPose(rel_dyn_pose_f.inverse());

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
    // Free up memory now that we've fused the frame!
    frame.instance_view.DiscardView();

    // TODO remove if unnecessary cleanup
//    track.SetNeedsCleanup(true);
//    // Free up memory from the previous frame.
//    int prev_idx = static_cast<int>(frame_idx) - 1;
//    if (prev_idx >= 0) {
//      auto prev_frame = track.GetFrame(prev_idx);
//      prev_frame.instance_view.DiscardView();
//    }
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

  // TODO(andrei): This is obviously wasteful!
  delete mesh;
  delete meshing_engine;
}

vector<InstanceView> InstanceReconstructor::CreateInstanceViews(
    const InstanceSegmentationResult &segmentation_result,
    ITMLib::Objects::ITMView *main_view,
    const SparseSceneFlow &scene_flow
) {
  Vector2i frame_size_itm = main_view->rgb->noDims;
  Eigen::Vector2i frame_size(frame_size_itm.x, frame_size_itm.y);

  vector<InstanceView> instance_views;
  for (const InstanceDetection &instance_detection : segmentation_result.instance_detections) {
    if (IsPossiblyDynamic(instance_detection.GetClassName())) {
      // bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
      // check this, since the field is private.
      bool use_gpu = true;

      // The ITMView takes ownership of this.
      ITMRGBDCalib *calibration = new ITMRGBDCalib;
      *calibration = *main_view->calib;
      auto view = make_shared<ITMView>(calibration, frame_size_itm, frame_size_itm, use_gpu);
      vector<RawFlow> instance_flow_vectors;
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
                                             vector<RawFlow> &out_instance_flow_vectors,
                                             const InstanceDetection &detection,
                                             const Eigen::Vector2i &frame_size,
                                             bool check_sf_start) {
  const BoundingBox &flow_bbox = detection.conservative_mask->GetBoundingBox();
  map<pair<int, int>, RawFlow> coord_to_flow;
  int frame_width = frame_size(0);
  int frame_height = frame_size(1);

  // Instead of expensively doing a per-pixel for every SF vector (ouch!), we just use the bounding
  // boxes, since we'll be using those vectors for RANSAC anyway. In the future, we could maybe
  // use some sort of hashing/sparse matrix for the scene flow and support per-pixel stuff.
  for(const auto &match : scene_flow.matches) {
    // TODO(andrei): Store old motion of car in track and use to initialize RANSAC under a constant
    // motion assumption (poor man's Kalman filtering).
    int fx = static_cast<int>(match.curr_left(0));
    int fy = static_cast<int>(match.curr_left(1));
    int fx_prev = static_cast<int>(match.prev_left(0));
    int fy_prev = static_cast<int>(match.prev_left(1));

    if (flow_bbox.ContainsPoint(fx, fy)) {
      // Use the larger mask so we only filter out truly ridiculous SF values
      if (!check_sf_start || detection.copy_mask->GetBoundingBox().ContainsPoint(fx_prev, fy_prev)) {
        coord_to_flow.emplace(pair<pair<int, int>, RawFlow>(pair<int, int>(fx, fy), match));
      }
    }
  }

  for (int cons_row = 0; cons_row < flow_bbox.GetHeight(); ++cons_row) {
    for (int cons_col = 0; cons_col < flow_bbox.GetWidth(); ++cons_col) {
      int cons_frame_row = cons_row + flow_bbox.r.y0;
      int const_frame_col = cons_col + flow_bbox.r.x0;

      if (cons_frame_row >= frame_height || const_frame_col >= frame_width) {
        continue;
      }

      u_char c_mask_val = detection.conservative_mask->GetData()->at<u_char>(cons_row, cons_col);
      if (c_mask_val == 1) {
        auto coord_pair = pair<int, int>(const_frame_col, cons_frame_row);
        if (coord_to_flow.find(coord_pair) != coord_to_flow.cend()) {
          out_instance_flow_vectors.push_back(coord_to_flow.find(coord_pair)->second);
        }
      }
    }
  }
}

}  // namespace reconstruction
}  // namespace instreclib
