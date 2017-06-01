#ifndef DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
#define DYNSLAM_PRECOMPUTEDDEPTHENGINE_H

#include "DepthEngine.h"

namespace dynslam {

class PrecomputedDepthEngine : public DepthEngine {

 public:
  void DisparityMapFromStereo(const cv::Mat &left,
                              const cv::Mat &right,
                              cv::Mat &out_disparity) override;

};

} // namespace dynslam

#endif //DYNSLAM_PRECOMPUTEDDEPTHENGINE_H
