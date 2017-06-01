

#include "DynSlam.h"
#include "Input.h"
#include "PrecomputedDepthEngine.h"

namespace dynslam {

using namespace instreclib::reconstruction;

void DynSlam::Initialize(InfiniTamDriver* itm_static_scene_engine_,
                         ImageSourceEngine* image_source,
                         SegmentationProvider* segmentation_provider) {

  this->image_source_ = image_source;

  // TODO make sure you pass this to the slam object. ideally, it shouldn't even touch it directly.
  string dir = "/home/barsana/datasets/kitti/odometry-dataset/sequences/06";
  this->input_ = new Input(
      dir,
      new PrecomputedDepthEngine(dir + "/precomputed-depth/Frames/", "%04d.pgm"),
      image_source->calib
  );

  window_size_.x = image_source->getDepthImageSize().x;
  window_size_.y = image_source->getDepthImageSize().y;

  this->static_scene_ = itm_static_scene_engine_;
  this->current_frame_no_ = 0;

  bool allocate_gpu = true;
  Vector2i input_shape = image_source->getDepthImageSize();
  out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
  out_image_float_ = new ITMFloatImage(input_shape, true, allocate_gpu);
  input_rgb_image_= new ITMUChar4Image(input_shape, true, allocate_gpu);
  input_raw_depth_image_ = new ITMShortImage(input_shape, true, allocate_gpu);

  // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.
  ITMSafeCall(cudaThreadSynchronize());

  this->segmentation_provider_ = segmentation_provider;
  this->instance_reconstructor_ = new InstanceReconstructor(itm_static_scene_engine_);

  cout << "DynSLAM initialization complete." << endl;
}

void DynSlam::ProcessFrame() {
//  if (! image_source_->hasMoreImages()) {
//    cout << "No more frames left in image source." << endl;
//    return;
//  }

  // Read the images from the first part of the pipeline
//  image_source_->getImages(input_rgb_image_, input_raw_depth_image_);
  if (! input_->HasMoreImages()) {
    cout << "No more frames left in image source." << endl;
    return;
  }

  input_->GetITMImages(input_rgb_image_, input_raw_depth_image_);

  static_scene_->UpdateView(input_rgb_image_, input_raw_depth_image_);

  // InstRec: semantic segmentation
  auto segmentationResult = segmentation_provider_->SegmentFrame(input_rgb_image_);
  cout << segmentationResult << endl;

  // Split the scene up into instances, and fuse each instance independently.
  instance_reconstructor_->ProcessFrame(static_scene_->GetView(), *segmentationResult);

  // Perform the tracking after the segmentation, so that we may in the future leverage semantic
  // information to enhance tracking.
  static_scene_->Track();
  static_scene_->Integrate();
  static_scene_->PrepareNextStep();

  ITMSafeCall(cudaThreadSynchronize());

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
