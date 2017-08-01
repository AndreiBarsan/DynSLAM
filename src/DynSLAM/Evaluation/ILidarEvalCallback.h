
#ifndef DYNSLAM_ILIDAREVALCALLBACK_H
#define DYNSLAM_ILIDAREVALCALLBACK_H

#include <Eigen/Eigen>

class ILidarEvalCallback {
 public:
  /// \brief Called for every Velodyne point which falls on the camera frame.
  virtual void ProcessItem(int idx,
                           const Eigen::Vector3d &velo_2d_homo,
                           unsigned char rendered_depth,
                           unsigned char input_depth,
                           unsigned char velodyne_depth,
                           int width,
                           int height) = 0;
};

#endif //DYNSLAM_ILIDAREVALCALLBACK_H
