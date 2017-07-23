

#include <chrono>
#include <thread>

#include "DynSlam.h"

namespace dynslam {

using namespace instreclib::reconstruction;
using namespace dynslam::eval;

void DynSlam::Initialize(InfiniTamDriver *itm_static_scene_engine,
                         SegmentationProvider *segmentation_provider,
                         SparseSFProvider *sparse_sf_provider,
                         Evaluation *evaluation) {

  this->static_scene_ = itm_static_scene_engine;
  this->sparse_sf_provider_ = sparse_sf_provider;
  this->segmentation_provider_ = segmentation_provider;
  this->evaluation_ = evaluation;
  this->current_frame_no_ = 0;

  bool allocate_gpu = true;

  Vector2i input_shape = itm_static_scene_engine->GetImageSize();
  out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
  out_image_float_ = new ITMFloatImage(input_shape, true, allocate_gpu);
  input_rgb_image_ = new cv::Mat3b(input_shape.x, input_shape.y);
  input_raw_depth_image_ = new cv::Mat1s(input_shape.x, input_shape.y);

  input_width_ = input_shape.x;
  input_height_ = input_shape.y;

  // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.

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

  future<shared_ptr<InstanceSegmentationResult>> seg_result_future = async(launch::async, [this] {
    utils::Tic("Semantic segmentation");
    auto segmentation_result = segmentation_provider_->SegmentFrame(*input_rgb_image_);
    utils::Toc();
    return segmentation_result;
  });

  future<void> tracking_and_ssf = async(launch::async, [this, &input, &first_frame] {
    utils::Tic("Visual Odometry");
    static_scene_->Track();
    utils::Toc();

    utils::Tic("Sparse Scene Flow");
    cv::Mat1b *left_gray, *right_gray;
    input->GetCvStereoGray(&left_gray, &right_gray);

    // TODO(andrei): Pass egomotion here, so that SF computation is relative to egomotion (so most
    // SF values, such as those associated with the road or buildings would become close to zero).
    // Look at libviso2 source code for inspiration on the relation between egomotion and SF. How do
    // they extract egomotion from the SF?
    sparse_sf_provider_->ComputeSparseSF(
        make_pair((cv::Mat1b *) nullptr, (cv::Mat1b *) nullptr),
        make_pair(left_gray, right_gray)
    );
    if (!sparse_sf_provider_->FlowAvailable() && !first_frame) {
      cerr << "Warning: could not compute scene flow." << endl;
    }
    utils::Toc();
  });

  seg_result_future.wait();
  tracking_and_ssf.wait();

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
    instance_reconstructor_->ProcessFrame(
        this,
        static_scene_->GetView(),
        *seg_result_future.get(),
        sparse_sf_provider_->GetFlow(),
        *sparse_sf_provider_,
        always_reconstruct_objects_);
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
    if (enable_map_decay_) {
      utils::Tic("Map decay");
      static_scene_->Decay();
      utils::TocMicro();
    }
  }

  // TODO(andrei): Easy way to toggle this on/off.
  utils::Tic("Evaluation");
  evaluation_->EvaluateFrame(input, this);
  utils::Toc();

  // Final sanity check after the frame is processed: individual components should check for errors.
  // If something slips through and gets here, it's bad and we want to stop execution.
  ITMSafeCall(cudaDeviceSynchronize());
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    cerr << "A CUDA error slipped undetected from a component of DynSLAM!" << endl;

    // Trigger the regular error response.
    ITMSafeCall(last_error);
  }

  current_frame_no_++;
}

const unsigned char* DynSlam::GetObjectPreview(int object_idx) {
  ITMUChar4Image *preview = instance_reconstructor_->GetInstancePreviewRGB(object_idx);
  if (nullptr == preview) {
    // This happens when there's no instances to preview.
    out_image_->Clear();
  } else {
    out_image_->SetFrom(preview, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
  }

  return out_image_->GetData(MemoryDeviceType::MEMORYDEVICE_CPU)->getValues();
}

void DynSlam::SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const {
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string map_fpath = utils::Format("%s/static-%s-mesh-%06d-frames.obj",
                                   target_folder.c_str(),
                                   depth_name.c_str(),
                                   current_frame_no_);
  cout << "Saving full static map to: " << map_fpath << endl;
  static_scene_->SaveSceneToMesh(map_fpath.c_str());
}

void DynSlam::SaveDynamicObject(const std::string &dataset_name,
                                const std::string &depth_name,
                                int object_id) const {
  cout << "Saving mesh for object #" << object_id << "'s reconstruction..." << endl;
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string instance_fpath = utils::Format("%s/instance-%s-%06d-mesh.obj",
                                        target_folder.c_str(),
                                        depth_name.c_str(),
                                        object_id);
  instance_reconstructor_->SaveObjectToMesh(object_id, instance_fpath);

  cout << "Done saving mesh for object #" << object_id << "'s reconstruction in file ["
       << instance_fpath << "]." << endl;
}

std::string DynSlam::EnsureDumpFolderExists(const string &dataset_name) const {
  // TODO-LOW(andrei): Make this more cross-platform and more secure.
  string today_folder = utils::GetDate();
  string target_folder = "mesh_out/" + dataset_name + "/" + today_folder;
  if(system(utils::Format("mkdir -p '%s'", target_folder.c_str()).c_str())) {
    throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
  }

  return target_folder;
}

}
