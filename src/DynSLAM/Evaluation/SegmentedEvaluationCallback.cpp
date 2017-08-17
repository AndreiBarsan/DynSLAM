#include "SegmentedEvaluationCallback.h"

namespace dynslam {
namespace eval {

using namespace instreclib::reconstruction;

void SegmentedEvaluationCallback::ProcessLidarPoint(int idx,
                                                    const Eigen::Vector3d &velo_2d_homo_px,
                                                    float rendered_disp,
                                                    float rendered_depth_m,
                                                    float input_disp,
                                                    float input_depth_m,
                                                    float lidar_disp,
                                                    int frame_width,
                                                    int frame_height) {

  int px = static_cast<int>(round(velo_2d_homo_px(0)));
  int py = static_cast<int>(round(velo_2d_homo_px(1)));

  bool matched_to_dyn_object = false;

  for (const InstanceDetection &det : frame_segmentation_->instance_detections) {
    // Use the mask used for feeding the reconstruction to establish if we're inside a dynamic
    // object.
    if (! det.copy_mask->ContainsPoint(px, py)) {
      continue;
    }

    if (InstanceReconstructor::IsPossiblyDynamic(det.GetClassName())) {
      matched_to_dyn_object = true;

      if (InstanceReconstructor::ShouldReconstruct(det.GetClassName())) {
        bool is_reconstructed = false;

        if(reconstructor_ != nullptr) {
          /// TODO-LOW(andrei): This is dirty, but it works... It should nevertheless be improved.
          const Track &track = reconstructor_->GetTrackAtPoint(px, py);
          is_reconstructed = track.GetState() != TrackState::kUncertain;
        }

        if (is_reconstructed) {
          dynamic_eval_.ProcessLidarPoint(idx,
                                          velo_2d_homo_px,
                                          rendered_disp,
                                          rendered_depth_m,
                                          input_disp,
                                          input_depth_m,
                                          lidar_disp,
                                          frame_width,
                                          frame_height);
        }
        else {
          // A car we aren't reconstructing, e.g. because it just entered the scene.
          // Do not evaluate this point.
          skipped_lidar_points_++;
        }
      } else {
        // A dynamic but non-reconstructable object, like a pedestrian.
        // Do not evaluate this point.
        skipped_lidar_points_++;
      }
    }
    else {
      // LIDAR point belongs to e.g., a plant or table.
      // We should consider this as part of the static map.
    }

    // Safe to break, since the masks are guaranteed never to overlap.
    break;
  }

  if (! matched_to_dyn_object) {
    static_eval_.ProcessLidarPoint(idx,
                                   velo_2d_homo_px,
                                   rendered_disp,
                                   rendered_depth_m,
                                   input_disp,
                                   input_depth_m,
                                   lidar_disp,
                                   frame_width,
                                   frame_height);
  }
}

}
}

