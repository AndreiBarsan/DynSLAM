
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

/// \brief Masks the scene flow using the (smaller) conservative mask.
void ExtractSceneFlow(
    const SparseSceneFlow &scene_flow,
    vector<RawFlow> &out_instance_flow_vectors,
    const InstanceDetection &detection,
    int frame_width,
    int frame_height,
    bool check_sf_start = true
) {
  const BoundingBox &flow_bbox = detection.conservative_mask->GetBoundingBox();
  map<pair<int, int>, RawFlow> coord_to_flow;

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
      if (!check_sf_start || detection.mask->GetBoundingBox().ContainsPoint(fx_prev, fy_prev)) {
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

      u_char c_mask_val = detection.conservative_mask->GetMaskData()->at<u_char>(cons_row, cons_col);
      if (c_mask_val == 1) {
        auto coord_pair = pair<int, int>(const_frame_col, cons_frame_row);
        if (coord_to_flow.find(coord_pair) != coord_to_flow.cend()) {
          out_instance_flow_vectors.push_back(coord_to_flow.find(coord_pair)->second);
        }
      }
    }
  }
}

// TODO(andrei): Implement this in CUDA. It should be easy.
template <typename DEPTH_T>
void ProcessSilhouette_CPU(Vector4u *sourceRGB,
                           DEPTH_T *sourceDepth,
                           Vector4u *destRGB,
                           DEPTH_T *destDepth,
                           Vector2i sourceDims,
                           vector<RawFlow> &instance_flow_vectors,  // expressed in the current left camera's frame
                           const InstanceDetection &detection,
                           const SparseSceneFlow &scene_flow) {
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

  int frame_width = sourceDims[0];
  int frame_height = sourceDims[1];
  const BoundingBox &bbox = detection.GetBoundingBox();

  int box_width = bbox.GetWidth();
  int box_height = bbox.GetHeight();

  memset(destRGB, 0, frame_width * frame_height * sizeof(*sourceRGB));
  memset(destDepth, 0, frame_width * frame_height * sizeof(DEPTH_T));

  // Keep track of the minimum depth in the frame, so we can use it as a heuristic when
  // reconstructing instances.
  float min_depth = numeric_limits<float>::max();
  const float kInvalidDepth = -1.0f;

  for (int row = 0; row < box_height; ++row) {
    for (int col = 0; col < box_width; ++col) {
      int frame_row = row + bbox.r.y0;
      int frame_col = col + bbox.r.x0;
      // TODO(andrei): Are the CPU-specific InfiniTAM functions doing this in a nicer way, or are
      // they also just looping?

      if (frame_row < 0 || frame_row >= frame_height ||
          frame_col < 0 || frame_col >= frame_width) {
        continue;
      }

      int frame_idx = frame_row * frame_width + frame_col;
      u_char mask_val = detection.mask->GetMaskData()->at<u_char>(row, col);
      if (mask_val == 1) {
        destRGB[frame_idx].r = sourceRGB[frame_idx].r;
        destRGB[frame_idx].g = sourceRGB[frame_idx].g;
        destRGB[frame_idx].b = sourceRGB[frame_idx].b;
        destRGB[frame_idx].a = sourceRGB[frame_idx].a;
        sourceRGB[frame_idx].r = 0;
        sourceRGB[frame_idx].g = 0;
        sourceRGB[frame_idx].b = 0;
        sourceRGB[frame_idx].a = 0;

        float depth = sourceDepth[frame_idx];
        destDepth[frame_idx] = depth;
        sourceDepth[frame_idx] = 0.0f;

        if (depth != kInvalidDepth && depth < min_depth) {
          min_depth = depth;
        }
      }
    }
  }

  ExtractSceneFlow(
      scene_flow,
      instance_flow_vectors,
      detection,
      frame_width,
      frame_height);

  // TODO(andrei): Store min depth somewhere.
  cout << "Instance frame min depth: " << min_depth << endl;
}

// TODO(andrei): Rename variable according to style guide.
template<typename TDepth>
void RemoveSilhouette_CPU(ORUtils::Vector4<unsigned char> *sourceRGB,
                          TDepth *sourceDepth,
                          ORUtils::Vector2<int> sourceDims,
                          const segmentation::InstanceDetection &detection) {

  int frame_width = sourceDims[0];
  int frame_height = sourceDims[1];
  const BoundingBox &bbox = detection.GetBoundingBox();

  int box_width = bbox.GetWidth();
  int box_height = bbox.GetHeight();

  for (int row = 0; row < box_height; ++row) {
    for (int col = 0; col < box_width; ++col) {
      int frame_row = row + bbox.r.y0;
      int frame_col = col + bbox.r.x0;
      // TODO(andrei): Are the CPU-specific InfiniTAM functions doing this in a nicer way, or are
      // they also just looping?

      if (frame_row < 0 || frame_row >= frame_height ||
          frame_col < 0 || frame_col >= frame_width) {
        continue;
      }

      int frame_idx = frame_row * frame_width + frame_col;
      u_char mask_val = detection.mask->GetMaskData()->at<u_char>(row, col);
      if (mask_val == 1) {
        sourceRGB[frame_idx].r = 0;
        sourceRGB[frame_idx].g = 0;
        sourceRGB[frame_idx].b = 0;
        sourceRGB[frame_idx].a = 0;
      }
    }
  }

}


Option<Eigen::Matrix4d>* EstimateInstanceMotion(
    const vector<RawFlow> &instance_raw_flow,
    const SparseSFProvider &ssf_provider
) {
  // This is a good conservative value, but we can definitely do better.
//  uint32_t kMinFlowVectorsForPoseEst = 25;   // TODO experiment and set proper value;
  uint32_t kMinFlowVectorsForPoseEst = 12;
  // technically 3 should be enough (because they're stereo-and-time 4-way correspondences, but
  // we're being a little paranoid).
  size_t flow_count = instance_raw_flow.size();

  // TODO(andrei): Consider doing these computations in the tracker, or lazily, only when the
  // system decides to start reconstructing something, maybe?

  // TODO(andrei): Since we're not doing egomotion computation, the nr of inputs is much lower
  // than in the full viso case => fewer RANSAC iterations than the default should be OK,
  // especially if we then use a direct method for refinement.
  if (instance_raw_flow.size() >= kMinFlowVectorsForPoseEst) {
    vector<double> instance_motion_delta = ssf_provider.ExtractMotion(instance_raw_flow);
    if (instance_motion_delta.size() != 6) {
      // track information not available yet; idea: we could move this computation into the
      // track object, and use data from many more frames (if available).
          cerr << "Could not compute instance delta motion from " << flow_count << " matches." << endl;
      return new Option<Eigen::Matrix4d>();
    } else {
          cout << "Successfully estimated the relative instance pose from " << flow_count
               << " matches." << endl;
      Matrix delta_mx = VisualOdometry::transformationVectorToMatrix(instance_motion_delta);

      // TODO(andrei): Prof Geiger: compare this to the current egomotion (or to ID if egomotion
      // removed). If close to egomotion (compare correctly! absolute translation l2 + rodrigues
      // rotation repr.), then the object is static; we set the relative motion to id and BAM! done.

      // TODO(andrei): Make this a utility. Note: The '~' transposes the matrix...
      auto *md_mat = new Eigen::Matrix4d((~delta_mx).val[0]);
      return new Option<Eigen::Matrix4d>(md_mat);
    }
  }
  else {
    cout << "Only " << flow_count << " scene flow points. Not estimating relative pose." << endl;
    return new Option<Eigen::Matrix4d>();
  }
}

void InstanceReconstructor::ProcessFrame(
    const dynslam::DynSlam* dyn_slam,
    ITMLib::Objects::ITMView *main_view,
    const segmentation::InstanceSegmentationResult &segmentation_result,
    const SparseSceneFlow &scene_flow,
    const SparseSFProvider &ssf_provider
) {
  // TODO(andrei): Perform this slicing 100% on the GPU.
  main_view->rgb->UpdateHostFromDevice();
  main_view->depth->UpdateHostFromDevice();

  ORUtils::Vector4<unsigned char> *rgb_data_h =
      main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

  cerr << "TODO: but why not at least attempt to perform motion analysis on these dudes, in case "
      "there's something like a stationary bike, or train, even if we never plan on attempting to "
      "reconstruct them... For persons, it might often fail, but in that case we may just default "
      "to deleting them out of the frame anyway..." << endl;
  // Note: for a real self-driving cars, you definitely want a completely generic obstacle detector.
  vector<string> classes_to_reconstruct_voc2012 = { "car" };
  vector<string> possibly_dynamic_classes_voc2012 = {
      "airplane",   // you never know...
      "bicycle",
      "bird",
      "boat",       // again, you never know...
      "bus",
      "car",
      "cat",
      "cow",
      "dog",
      "horse",
      "motorbike",
      "person",
      "sheep",
      "train"
  };

  vector<InstanceView> new_instance_views;
  for (const InstanceDetection &instance_detection : segmentation_result.instance_detections) {
    // At this stage of the project, we only care about cars. In the future, this scheme could be
    // extended to also support other classes, as well as any unknown, but moving, objects.
    if (find(classes_to_reconstruct_voc2012.begin(),
             classes_to_reconstruct_voc2012.end(),
             instance_detection.GetClassName()) != classes_to_reconstruct_voc2012.end()
    ) {
      Vector2i frame_size = main_view->rgb->noDims;
      // bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
      // check this, since the field is private.
      bool use_gpu = true;

      // The ITMView takes ownership of this.
      ITMRGBDCalib *calibration = new ITMRGBDCalib;
      *calibration = *main_view->calib;

      auto view = make_shared<ITMView>(calibration, frame_size, frame_size, use_gpu);
      auto rgb_segment_h = view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
      auto depth_segment_h = view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

      vector<RawFlow> instance_raw_flow;
      ProcessSilhouette_CPU(rgb_data_h,
                            depth_data_h,
                            rgb_segment_h,
                            depth_segment_h,
                            main_view->rgb->noDims,
                            instance_raw_flow,
                            instance_detection,
                            scene_flow);

      view->rgb->UpdateDeviceFromHost();
      view->depth->UpdateDeviceFromHost();

      Option<Eigen::Matrix4d> *motion_delta = EstimateInstanceMotion(instance_raw_flow, ssf_provider);

      new_instance_views.emplace_back(instance_detection,
                                      view,
                                      instance_raw_flow,
                                      shared_ptr<Option<Eigen::Matrix4d>>(motion_delta));
    }
    else if(find(possibly_dynamic_classes_voc2012.cbegin(),
                 possibly_dynamic_classes_voc2012.cend(),
                 instance_detection.GetClassName()) != possibly_dynamic_classes_voc2012.cend()
    ) {
      // In this case, we simply remove the object from view, but we don't attempt to track or
      // reconstruct it.
      RemoveSilhouette_CPU(rgb_data_h, depth_data_h, main_view->rgb->noDims, instance_detection);
    }
    // else, it's not an object in which we are interested
  }

  // Associate this frame's detection(s) with those from previous frames.
  this->instance_tracker_->ProcessInstanceViews(frame_idx_, new_instance_views, dyn_slam->GetPose());
  this->ProcessReconstructions();

  // Update the GPU image after we've (if applicable) removed the dynamic objects from it.
  main_view->rgb->UpdateDeviceFromHost();
  main_view->depth->UpdateDeviceFromHost();

  // ``Graphically'' display the object tracks for debugging.
  /*
  for (const auto &pair: this->instance_tracker_->GetActiveTracks()) {
    cout << "Track: " << pair.second.GetAsciiArt() << endl;
  }
  // */

  frame_idx_++;
}

ITMUChar4Image *InstanceReconstructor::GetInstancePreviewRGB(size_t track_idx) {
  if (! instance_tracker_->HasTrack(track_idx)) {
    return nullptr;
  }

  return instance_tracker_->GetTrack(track_idx).GetLastFrame().instance_view.GetView()->rgb;
}

ITMFloatImage *InstanceReconstructor::GetInstancePreviewDepth(size_t track_idx) {
  const auto &tracks = instance_tracker_->GetActiveTracks();
  if (tracks.empty()) {
    return nullptr;
  }

  size_t idx = track_idx;
  if (idx >= tracks.size()) {
    idx = tracks.size() - 1;
  }

  return tracks.at(idx).GetLastFrame().instance_view.GetView()->depth;
}

void InstanceReconstructor::ProcessReconstructions() {
  // TODO loop through keys only since we want to do all track accesses through the instance tracker for constness reasons
  for (auto &pair : instance_tracker_->GetActiveTracks()) {
    Track& track = instance_tracker_->GetTrack(pair.first);

    if(track.GetLastFrame().frame_idx != frame_idx_) {
       //        (track.GetId() != 3 && track.GetId() != 0 && track.GetId() != 1)) {
      // If we don't have any new information in this track, there's nothing to fuse.
      continue;
    }

    if (! track.HasReconstruction()) {
      // TODO(andrei): Consider comparing object motion relative to camera---if close to none, then
      // no point in reconstructing, and we can just merge the object into the map.
      bool eligible = track.EligibleForReconstruction();

      if (! eligible) {
        // The frame data we have is insufficient, so we won't try to reconstruct the object
        // (yet).
        continue;
      }

      // No reconstruction allocated yet; let's initialize one.
      cout << endl << endl;
      cout << "Starting to reconstruct instance with ID: " << track.GetId() << endl << endl;
      ITMLibSettings *settings = new ITMLibSettings(*driver->GetSettings());

      // Set a much smaller voxel block number for the reconstruction, since individual objects
      // occupy a limited amount of space in the scene.
      // TODO(andrei): Set this limit based on some physical specification, such as 10m x 10m x
      // 10m.
//      settings->sdfLocalBlockNum = 2500;
      settings->sdfLocalBlockNum = 10000;
      // We don't want to create an (expensive) meshing engine for every instance.
      settings->createMeshingEngine = false;

      track.GetReconstruction() = make_shared<InfiniTamDriver>(
          settings,
          driver->GetView()->calib,
          driver->GetView()->rgb->noDims,
          driver->GetView()->rgb->noDims);

      // If we already have some frames, integrate them into the new volume.
      for(size_t i = 0; i < track.GetSize(); ++i) {
        // TODO(andrei): This accounts for gaps in tracks for static objects, but not for dynamic ones.
        FuseFrame(track, i);
      }
    } else {
      cout << "Continuing to reconstruct instance with ID: " << track.GetId() << endl;
      // Fuse the latest frame into the volume.
      FuseFrame(track, track.GetSize() - 1);
    }

  }
}

void InstanceReconstructor::FuseFrame(Track &track, size_t frame_idx) const {
  InfiniTamDriver &instance_driver = *track.GetReconstruction();

  TrackFrame &frame = track.GetFrame(frame_idx);
  instance_driver.SetView(frame.instance_view.GetView());
  Option<Eigen::Matrix4d> rel_dyn_pose = track.GetFramePose(frame_idx);

  // Only fuse the information if the relative pose could be established.
  // TODO(andrei): This would require modifying libviso a little, but in the even that we miss a
  // frame, i.e., we are unable to estimate the relative pose from frame k to frame k+1, we could
  // still try to estimate it from k to k+2.
  if (rel_dyn_pose.IsPresent()) {
    Eigen::Matrix4f rel_dyn_pose_f = (*rel_dyn_pose).cast<float>();

    cout << "Fusing frame " << frame_idx << "/ #" << track.GetId() << "." << endl << rel_dyn_pose_f << endl;

    // TODO(andrei): Replace this with fine tracking initialized by this coarse relative pose.
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
//      cout << "Finished instance integration." << endl << endl;
  }
  else {
    cout << "Could not fuse instance data for track #" << track.GetId() << " due to missing pose "
         << "information." << endl;
  }
}

void InstanceReconstructor::SaveObjectToMesh(int object_id, const string &fpath) {
  // TODO nicer error handling
  if(! instance_tracker_->HasTrack(object_id)) {
    throw std::runtime_error("Unknown track");
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

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cerr << "Warning: we seem to have inherited an error here. Meshing should work OK but you "
         << "should look into this..." << endl;
  }

  meshing_engine->MeshScene(mesh, track.GetReconstruction()->GetScene());
  mesh->WriteOBJ(fpath.c_str());
//    mesh->WriteSTL(fpath.c_str());

  // TODO(andrei): This is obviously wasteful!
  delete mesh;
  delete meshing_engine;
}

}  // namespace reconstruction
}  // namespace instreclib
