
#include "SegmentedEvaluationCallback.h"
#include "SegmentedCallback.h"


DECLARE_int32(fusion_every);

namespace dynslam {
namespace eval {

/// \brief Establishes if the given LIDAR point corresponds to a static part of the input, to a
///        dynamic part being reconstructed by DynSLAM, or neither.
SegmentedCallback::LidarAssociation SegmentedCallback::GetPointAssociation(
    const Eigen::Vector3d &velo_2d_homo_px
) {
  int px = static_cast<int>(round(velo_2d_homo_px(0)));
  int py = static_cast<int>(round(velo_2d_homo_px(1)));

  for (const InstanceDetection &det : frame_segmentation_->instance_detections) {
    // Use the mask used for generating the reconstruction to establish if we're inside a dynamic
    // object.
    if (!det.copy_mask->ContainsPoint(px, py)) {
      continue;
    }

    if (InstanceReconstructor::IsPossiblyDynamic(det.GetClassName())) {

      if (InstanceReconstructor::ShouldReconstruct(det.GetClassName())) {
        bool is_reconstructed = false;

        // Does not support dynamic object reconstruction evaluation when skipping frames.
        if (reconstructor_ != nullptr && FLAGS_fusion_every == 1)
        {
          /// TODO-LOW(andrei): This is dirty, but it works... It should nevertheless be improved.
          const Track &track = reconstructor_->GetTrackAtPoint(px, py);
          is_reconstructed = track.GetState() != kUncertain;
        }

        if (is_reconstructed) {
          return kDynamicReconstructed;
        } else {
          // A car we aren't reconstructing, e.g. because it just entered the scene.
          // Do not evaluate this point.
          skipped_lidar_points_++;
          return kNeither;
        }
      } else {
        // A dynamic but non-reconstructable object, like a pedestrian.
        // Do not evaluate this point.
        skipped_lidar_points_++;
        return kNeither;
      }
    } else {
      // LIDAR point belongs to e.g., a plant or table.
      // We should consider this as part of the static map.
    }

    // Safe to break, since the masks are guaranteed never to overlap.
    break;
  }

  // We're not part of a dynamic object, reconstructed or non-reconstructed, so we're static.
  return kStaticMap;
}

}
}

