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

  for (const InstanceDetection &det : frame_segmentation->instance_detections) {
    if (det.copy_mask->ContainsPoint(px, py)) {
      // Use the mask used for feeding the reconstruction to establish if we're inside a dynamic
      // object...
      if (InstanceReconstructor::IsPossiblyDynamic(det.GetClassName())) {
        ComputeAccuracy(rendered_disp, rendered_depth_m, input_disp, input_depth_m, lidar_disp,
                        input_stats_dynamic_, rendered_stats_dynamic_);
      }
    }
    else if (! det.delete_mask->ContainsPoint(px, py)) {
      // ...and the larger delete mask to consider anything outside it as part of the static map.
      ComputeAccuracy(rendered_disp, rendered_depth_m, input_disp, input_depth_m, lidar_disp,
                      input_stats_static_, rendered_stats_static_);
    }
  }

}

DepthEvaluation SegmentedEvaluationCallback::GetEvaluation() {
  // This looks like a sign we should use composition, not inheritance.
  throw runtime_error("Please use 'GetStaticEvaluation' or 'GetDynamicEvaluation'.");
}

}
}

