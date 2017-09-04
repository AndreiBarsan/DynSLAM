

#include <chrono>
#include <thread>

#include "DynSlam.h"
#include "Evaluation/Evaluation.h"

DEFINE_bool(dynamic_weights, false, "Whether to use depth-based weighting when performing fusion.");
DECLARE_bool(semantic_evaluation);
DECLARE_int32(evaluation_delay);

namespace dynslam {

using namespace instreclib::reconstruction;
using namespace dynslam::eval;

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
    if (dynamic_mode_ || FLAGS_semantic_evaluation) {
      utils::Timer timer("Semantic segmentation");
      timer.Start();
      auto segmentation_result = segmentation_provider_->SegmentFrame(*input_rgb_image_);
      timer.Stop();
      cout << timer.GetName() << " took " << timer.GetDuration() / 1000 << "ms" << endl;
      return segmentation_result;
    }
    else {
      return shared_ptr<InstanceSegmentationResult>(nullptr);
    }
  });

  future<void> ssf_and_vo = async(launch::async, [this, &input, &first_frame] {
    utils::Tic("Sparse Scene Flow");

    // Whether to use input from the original cameras. Unavailable with the tracking dataset.
    // When original gray images are not used, the color ones are converted to grayscale and passed
    // to the visual odometry instead.
    bool original_gray = false;

    // TODO-LOW(andrei): Reuse these buffers for performance.
    cv::Mat1b *left_gray, *right_gray;
    if (original_gray) {
      input->GetCvStereoGray(&left_gray, &right_gray);
    }
    else {
      cv::Mat3b *left_col, *right_col;
      input->GetCvStereoColor(&left_col, &right_col);

      left_gray = new cv::Mat1b(left_col->rows, left_col->cols);
      right_gray = new cv::Mat1b(right_col->rows, right_col->cols);

      cv::cvtColor(*left_col, *left_gray, cv::COLOR_RGB2GRAY);
      cv::cvtColor(*right_col, *right_gray, cv::COLOR_RGB2GRAY);
    }

    // TODO(andrei): Idea: compute only matches here, the make the instance reconstructor process
    // the frame and remove clearly-dynamic SF vectors (e.g., from tracks which are clearly dynamic,
    // as marked from a prev frame), before computing the egomotion, and then processing the
    // reconstructions. This may improve VO accuracy, and it could give us an excuse to also
    // evaluate ATE and compare it with the results from e.g., StereoScan, woo!
    sparse_sf_provider_->ComputeSparseSF(
        make_pair((cv::Mat1b *) nullptr, (cv::Mat1b *) nullptr),
        make_pair(left_gray, right_gray)
    );
    if (!sparse_sf_provider_->FlowAvailable() && !first_frame) {
      cerr << "Warning: could not compute scene flow." << endl;
    }
    utils::Toc("Sparse Scene Flow", false);

    utils::Tic("Visual Odometry");
    Eigen::Matrix4f delta = sparse_sf_provider_->GetLatestMotion();

    // TODO(andrei): Nicer way to do this switch.
    bool external_odo = true;
    if (external_odo) {
      Eigen::Matrix4f new_pose = delta * pose_history_[pose_history_.size() - 1];
      static_scene_->SetPose(new_pose.inverse());
      pose_history_.push_back(new_pose);
    }
    else {
      // Used when we're *not* computing VO as part of the SF estimation process.
      static_scene_->Track();
      Eigen::Matrix4f new_pose = static_scene_->GetPose();
      pose_history_.push_back(new_pose);
    }

    if (! original_gray) {
      delete left_gray;
      delete right_gray;
    }

    utils::Toc("Visual Odometry", false);
  });

  seg_result_future.wait();
  // 'get' ensures any exceptions are propagated (unlike 'wait').
  ssf_and_vo.get();

  utils::Tic("Input preprocessing");
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
  utils::Toc();

  // Split the scene up into instances, and fuse each instance independently.
  utils::Tic("Instance tracking and reconstruction");
  if (sparse_sf_provider_->FlowAvailable()) {
    this->latest_seg_result_ = seg_result_future.get();
    // We need flow information in order to correctly determine which objects are moving, so we
    // can't do this when no scene flow is available (i.e., in the first frame).
    if (dynamic_mode_ && current_frame_no_ % experimental_fusion_every_ == 0) {
      instance_reconstructor_->ProcessFrame(
          this,
          static_scene_->GetView(),
          *latest_seg_result_,
          sparse_sf_provider_->GetFlow(),
          *sparse_sf_provider_,
          always_reconstruct_objects_);
    }
  }
  utils::Toc();

  // Perform the tracking after the segmentation, so that we may in the future leverage semantic
  // information to enhance tracking.
  if (! first_frame) {
    if (current_frame_no_ % experimental_fusion_every_ == 0) {
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
  }

  int evaluated_frame_idx = current_frame_no_ - FLAGS_evaluation_delay;
  if (FLAGS_enable_evaluation && evaluated_frame_idx > 0) {
    utils::Tic("Evaluation");
    bool enable_compositing = (FLAGS_evaluation_delay == 0);
    evaluation_->EvaluateFrame(input, this, evaluated_frame_idx, enable_compositing);
    utils::Toc();
  }
  evaluation_->LogMemoryUse(this);

  // Final sanity check after the frame is processed: individual components should check for errors.
  // If something slips through and gets here, it's bad and we want to stop execution.
  ITMSafeCall(cudaDeviceSynchronize());
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    cerr << "A CUDA error slipped by undetected from a component of DynSLAM!" << endl;

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
