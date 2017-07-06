
#include "Evaluation.h"

namespace dynslam {
namespace eval {

void Evaluation::EvaluateFrame(Input *input, DynSlam *dyn_slam) {
  velodyne->ReadFrame(dyn_slam->GetCurrentFrameNo());

}

}
}
