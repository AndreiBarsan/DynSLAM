

#include <chrono>
#include <thread>

#include "DynSlam.h"
#include "Input.h"
#include "PrecomputedDepthProvider.h"

namespace dynslam {

using namespace instreclib::reconstruction;

void DynSlam::Initialize(InfiniTamDriver *itm_static_scene_engine,
                         SegmentationProvider *segmentation_provider,
                         SparseSFProvider *sparse_sf_provider) {

  this->static_scene_ = itm_static_scene_engine;
  this->sparse_sf_provider_ = sparse_sf_provider;
  this->current_frame_no_ = 0;

  bool allocate_gpu = true;

  Vector2i input_shape = itm_static_scene_engine->GetImageSize();
  out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
  input_rgb_image_ = new cv::Mat3b(input_shape.x, input_shape.y);
  input_raw_depth_image_ = new cv::Mat1s(input_shape.x, input_shape.y);

  input_width_ = input_shape.x;
  input_height_ = input_shape.y;

  // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.
  ITMSafeCall(cudaThreadSynchronize());

  this->segmentation_provider_ = segmentation_provider;
  this->instance_reconstructor_ = new InstanceReconstructor(itm_static_scene_engine);

  cout << "DynSLAM initialization complete." << endl;
}

void DynSlam::ProcessFrame(Input *input) {
  // Read the images from the first part of the pipeline
  if (! input->HasMoreImages()) {
    cout << "No more frames left in image source." << endl;
    return;
  }

  bool first_frame = (current_frame_no_ == 0);

  utils::Tic("Read input and compute depth");
  if(!input->ReadNextFrame()) {
    throw runtime_error("Could not read input from the data source.");
  }
  utils::Toc();

  cv::Mat1b *left_gray, *right_gray;
  input->GetCvStereoGray(&left_gray, &right_gray);

  utils::Tic("Semantic segmentation");
  auto segmentationResult = segmentation_provider_->SegmentFrame(*input_rgb_image_);
  utils::Toc();

  utils::Tic("Visual Odometry");
  static_scene_->Track();
  utils::Toc();

  utils::Tic("Sparse Scene Flow");
  // TODO(andrei): Pass egomotion here, so that SF computation is relative to egomotion (so most
  // SF values, such as those associated with the road or buildings would become close to zero).
  // Look at libviso2 source code for inspiration on the relation between egomotion and SF. How do
  // they extract egomotion from the SF?
//  sparse_sf_provider_->ComputeSparseSF(
//      make_pair((cv::Mat1b *) nullptr, (cv::Mat1b *) nullptr),
//      make_pair(left_gray, right_gray)
//  );
//  if (!sparse_sf_provider_->FlowAvailable() && !first_frame) {
//    cerr << "Warning: could not compute scene flow." << endl;
//  }
  utils::Toc();

  utils::Tic("Input preprocessing");
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
  utils::Toc();

  // Split the scene up into instances, and fuse each instance independently.
  utils::Tic("Instance tracking and reconstruction");
  if (sparse_sf_provider_->FlowAvailable()) {
    // We need flow information in order to correctly determine which objects are moving, so we
    // can't do this when no scene flow is available (i.e., in the first frame, unless an error
    // occurs).
//    instance_reconstructor_->ProcessFrame(
//        this,
//        static_scene_->GetView(),
//        *segmentationResult,
//        sparse_sf_provider_->GetFlow(),
//        *sparse_sf_provider_);
  }
  utils::Toc();

  // Perform the tracking after the segmentation, so that we may in the future leverage semantic
  // information to enhance tracking.
  if (! first_frame) {
    utils::Tic("Static map fusion");
    static_scene_->Integrate();
    static_scene_->PrepareNextStep();
    utils::TocMicro();

    // Idea: trigger decay not based on frame gap, but using translation-based threshold.
    // Decay old, possibly noisy, voxels to improve map quality and reduce its memory footprint.
    utils::Tic("Map decay");
    static_scene_->Decay();
    utils::TocMicro();
  }

//  utils::Tic("Final error check");
  // Final sanity check after the frame is processed: individual components should check for errors.
  // If something slips through and gets here, it's bad and we want to stop execution.
  ITMSafeCall(cudaDeviceSynchronize());
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    cerr << "A CUDA error slipped undetected from a component of DynSLAM!" << endl;

    // Trigger the regular error response.
    ITMSafeCall(last_error);
  }
//  utils::Toc();

  current_frame_no_++;
}

const unsigned char* DynSlam::GetObjectPreview(int object_idx) {
  ITMUChar4Image *preview = instance_reconstructor_->GetInstancePreviewRGB(object_idx);
  if (nullptr == preview) {
    // This happens when there's no instances to preview.
    out_image_->Clear();
//    out_image_float_->Clear();
  } else {
    out_image_->SetFrom(preview, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
//    out_image_float_->SetFrom(preview, ORUtils::MemoryBlock<float>::CPU_TO_CPU);
  }

//  return out_image_float_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  return out_image_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU)->getValues();
}

}
