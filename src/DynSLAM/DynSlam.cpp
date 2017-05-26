

#include "DynSlam.h"

namespace dynslam {

using namespace InstRecLib::Reconstruction;

void DynSlam::Initialize(InfiniTamDriver *itm_static_scene_engine_, ImageSourceEngine *image_source) {

  this->image_source_ = image_source;

  window_size_.x = image_source->getDepthImageSize().x;
  window_size_.y = image_source->getDepthImageSize().y;

  this->itm_static_scene_engine_ = itm_static_scene_engine_;
  this->current_frame_no_ = 0;

  bool allocate_gpu = true;
  Vector2i input_shape = image_source->getDepthImageSize();
  out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
  input_rgb_image_= new ITMUChar4Image(input_shape, true, allocate_gpu);
  input_raw_depth_image_ = new ITMShortImage(input_shape, true, allocate_gpu);

  // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.
  ITMSafeCall(cudaThreadSynchronize());

  cout << "DynSLAM initialization complete." << endl;
}

void DynSlam::ProcessFrame() {
  if (! image_source_->hasMoreImages()) {
    cout << "No more frames left in image source." << endl;
    return;
  }

  // Read the images from the first part of the pipeline
  image_source_->getImages(input_rgb_image_, input_raw_depth_image_);

  // Forward them to InfiniTAM for the background reconstruction.
  itm_static_scene_engine_->ProcessFrame(input_rgb_image_, input_raw_depth_image_);

  ITMSafeCall(cudaThreadSynchronize());

  current_frame_no_++;
}

const unsigned char* DynSlam::GetObjectPreview(int object_idx) {
  ITMUChar4Image *preview = itm_static_scene_engine_->GetInstanceReconstructor()->GetInstancePreviewRGB(object_idx);
  if (nullptr == preview) {
    // This happens when there's no instances to preview.
    out_image_->Clear();
  } else {
    out_image_->SetFrom(preview, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
  }

  return out_image_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU)->getValues();
}

}