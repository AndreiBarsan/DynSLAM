
#ifndef DYNSLAM_ILIDAREVALCALLBACK_H
#define DYNSLAM_ILIDAREVALCALLBACK_H

#include <Eigen/Eigen>

/// \brief Used for auxiliary tasks during evaluation.
class ILidarEvalCallback {
 public:
  /// \brief Called for every Velodyne point which falls on the camera frame.
  virtual void LidarPoint(int idx,
                          const Eigen::Vector3d &velo_2d_homo,
                          int rendered_disp,
                          float rendered_depth,
                          int input_disp,
                          float input_depth,
                          int velodyne_disp,
                          int width,
                          int height) = 0;
};

#endif //DYNSLAM_ILIDAREVALCALLBACK_H
