//
// Created by barsana on 6/1/17.
//

#include "PrecomputedDepthEngine.h"

namespace dynslam {

void PrecomputedDepthEngine::DisparityMapFromStereo(const cv::Mat &left,
                                                    const cv::Mat &right,
                                                    cv::Mat &out_disparity
) {
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.

}

}
