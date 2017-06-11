

#include "InstanceReconstructor.h"
#include "InstanceView.h"

#include <vector>

namespace instreclib {
namespace reconstruction {

using namespace std;
using namespace instreclib::segmentation;
using namespace instreclib::utils;
using namespace ITMLib::Objects;

// TODO(andrei): Implement this in CUDA. It should be easy.
template <typename DEPTH_T>
void ProcessSilhouette_CPU(Vector4u *sourceRGB, DEPTH_T *sourceDepth, Vector4u *destRGB,
                           DEPTH_T *destDepth, Vector2i sourceDims,
                           const InstanceDetection &detection) {
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

  for (int row = 0; row < box_height; ++row) {
    for (int col = 0; col < box_width; ++col) {
      int frame_row = row + bbox.r.y0;
      int frame_col = col + bbox.r.x0;
      // TODO(andrei): Are the CPU-specific InfiniTAM functions doing this in a
      // nicer way?

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

        destDepth[frame_idx] = sourceDepth[frame_idx];
        sourceDepth[frame_idx] = 0.0f;
      }
    }
  }
}

void InstanceReconstructor::ProcessFrame(
    ITMLib::Objects::ITMView *main_view,
    const segmentation::InstanceSegmentationResult &segmentation_result) {
  // TODO(andrei): Perform this slicing 100% on the GPU.
  main_view->rgb->UpdateHostFromDevice();
  main_view->depth->UpdateHostFromDevice();

  ORUtils::Vector4<unsigned char> *rgb_data_h =
      main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

  vector<InstanceView> new_instance_views;
  for (const InstanceDetection &instance_detection : segmentation_result.instance_detections) {
    // At this stage of the project, we only care about cars. In the future, this scheme could be
    // extended to also support other classes, as well as any unknown, but moving, objects.
    if (instance_detection.class_id == kPascalVoc2012.label_to_id.at("car")) {
      Vector2i frame_size = main_view->rgb->noDims;
      // bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
      // check this, since the field is private.
      bool use_gpu = true;

      // TODO(andrei): Release these objects! They leak now.
      ITMRGBDCalib *calibration = new ITMRGBDCalib;
      *calibration = *main_view->calib;

      auto view = make_shared<ITMView>(calibration, frame_size, frame_size, use_gpu);
      auto rgb_segment_h = view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
      auto depth_segment_h = view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

      ProcessSilhouette_CPU(rgb_data_h, depth_data_h, rgb_segment_h, depth_segment_h,
                            main_view->rgb->noDims, instance_detection);

      view->rgb->UpdateDeviceFromHost();
      view->depth->UpdateDeviceFromHost();

      new_instance_views.emplace_back(instance_detection, view);
    }
  }

  // Associate this frame's detection(s) with those from previous frames.
  this->instance_tracker_->ProcessInstanceViews(frame_idx_, new_instance_views);
  this->ProcessReconstructions();

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

    if( track.GetLastFrame().frame_idx != frame_idx_) {
      // If we don't have any new information in this track, there's nothing to fuse.
      continue;
    }

    if (! track.HasReconstruction()) {
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
      // Make the ground truth tracker start from the current frame, and not from the default
      // 0th frame.
      settings->groundTruthPoseOffset += track.GetStartTime();
      // TODO(andrei): Do the same once you support proper tracking, since you will need to
      // initialize the instance's "tracker" with some pose, or change the tracker used, etc.

      // Lowering this can slightly increase the quality of the object's reconstruction, but at the
      // cost of additional memory.
//      settings->sceneParams.voxelSize = 0.0025f;

      track.GetReconstruction() = make_shared<InfiniTamDriver>(
          settings,
          driver->GetView()->calib,
          driver->GetView()->rgb->noDims,
          driver->GetView()->rgb->noDims);

      // If we already have some frames, integrate them into the new volume.
      for(int i = 0; i < static_cast<int>(track.GetSize()) - 1; ++i) {
        TrackFrame &frame = track.GetFrame(i);
        InfiniTamDriver &reconstruction = *(track.GetReconstruction());

        reconstruction.SetView(frame.instance_view.GetView());
        // TODO(andrei): Account for gaps in the track!
        reconstruction.Track();

        try {
          reconstruction.Integrate();
        }
        catch(std::runtime_error &error) {
          // TODO(andrei): Custom dynslam allocation exception we can catch here to avoid fatal
          // errors.
          // This happens when we run out of memory on the GPU for this volume. We should prolly
          // have a custom exception/error code for this.
          cerr << "Caught runtime error while integrating new data into an instance volume: "
               << error.what() << endl << "Will continue regular operation." << endl;
        }

        reconstruction.PrepareNextStep();
      }
    } else {
      cout << "Continuing to reconstruct instance with ID: " << track.GetId() << endl;
    }

    // We now fuse the current frame into the reconstruction volume.
    InfiniTamDriver &instance_driver = *track.GetReconstruction();
    instance_driver.SetView(track.GetLastFrame().instance_view.GetView());

    // TODO(andrei): Figure out a good estimate for the coord frame for the object.
    // TODO(andrei): This seems like the place to shove in the scene flow data.

    cout << endl << endl << "Start instance integration for #" << track.GetId() << endl;

    // TODO(andrei): We shouldn't do any tracking inside the instances IMHO.
    cerr << "Not accounting for gaps in track!" << endl;
    instance_driver.Track();

    try {
      // TODO(andrei): See above and also fix here.
      instance_driver.Integrate();
    }
    catch(std::runtime_error &error) {
      cerr << "Caught runtime error while integrating new data into an instance volume: "
           << error.what() << endl << "Will continue regular operation." << endl;
    }

    instance_driver.PrepareNextStep();

    cout << "Finished instance integration." << endl << endl;
  }
}

}  // namespace reconstruction
}  // namespace instreclib
