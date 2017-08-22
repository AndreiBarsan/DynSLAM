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

  LidarAssociation association = GetPointAssociation(velo_2d_homo_px);
  if (association == kDynamicReconstructed) {
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
  else if (association == kStaticMap) {
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

