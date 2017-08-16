
#ifndef DYNSLAM_ILIDAREVALCALLBACK_H
#define DYNSLAM_ILIDAREVALCALLBACK_H

#include <Eigen/Eigen>

/// \brief Used for auxiliary tasks during evaluation.
class ILidarEvalCallback {
 public:

  virtual ~ILidarEvalCallback() = default;

  /// \brief Called for every Velodyne point which falls on the camera frame.
  virtual void ProcessLidarPoint(int idx,
                                 const Eigen::Vector3d &velo_2d_homo,
                                 float rendered_disp,
                                 float rendered_depth,
                                 float input_disp,
                                 float input_depth,
                                 float velodyne_disp,
                                 int frame_width,
                                 int frame_height) = 0;
};

#endif //DYNSLAM_ILIDAREVALCALLBACK_H
