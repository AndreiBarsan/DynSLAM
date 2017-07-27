
#include "Evaluation.h"

namespace dynslam {
namespace eval {

void Evaluation::EvaluateFrame(Input *input, DynSlam *dyn_slam) {
  velodyne->ReadFrame(input->GetCurrentFrame() - 1);

  // TODO(andrei): Raycast of static-only raycast vs. no-dynamic-mapping vs. lidar vs. depth map.
  // TODO(andrei): Raycast of static+dynamic raycast vs. no-dynamic-mapping vs. lidar vs. depth map
  // No-dynamic-mapping == our system but with no semantics, object tracking, etc., so just ITM on
  // stereo.

  // From the incremental dense semantic stereo ITM paper:
  /*
   * In order to evaluate accuracy, we follow the approach of Sengupta et al. [4], who measure
   * the number of pixels whose distance (in terms of depth) from the ground truth (in our case
   * the Velodyne data) after projection to the image plane is less than a fixed threshold.
   */
  // => TODO(andrei): Project map (or use current live raycast?) and compare to LIDAR.
  // => TODO(andrei): Plot error as fn of delta, the allowed error. (see Sengupta/Torr paper Fig 10).

}

}
}
