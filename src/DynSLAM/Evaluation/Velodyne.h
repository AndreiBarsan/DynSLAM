
#ifndef DYNSLAM_VELODYNE_H
#define DYNSLAM_VELODYNE_H

#include <string>

#include <Eigen/Eigen>
#include <fstream>
#include "../Defines.h"
#include "../Utils.h"

namespace dynslam {
namespace eval {

class Velodyne {
 public:
  using LidarReadings = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;

  /// \brief Each Velodyne point has 4 entries: X, Y, Z, and reflectance.
  static const unsigned short kMeasurementsPerPoint = 4;

  /// \brief The size of the buffer into which we initially read the data.
  static const size_t kBufferSize = 1000000;

  /// \brief 4x4 matrix which transforms 3D homogeneous coordinates from the Velodyne LIDAR's
  ///        coordinate frame to the camera's coordinate frame.
  const Eigen::Matrix4f velodyne_to_rgb;

  /// \brief 3x4 matrix which projects 3D homogeneous coordinates in the camera's coordinate
  ///        frame to 2D homogeneous coordinates expressed in pixels.
  const Eigen::MatrixXf rgb_project;

 private:
  const std::string folder_;
  const std::string fname_format_;
  float * const data_buffer_;
  float * latest_frame_;
  size_t latest_point_count_;

 public:
  SUPPORT_EIGEN_FIELDS;

  Velodyne(const std::string &folder_, const std::string &fname_format_,
           const Eigen::Matrix4f &velodyne_to_rgb, const Eigen::MatrixXf &rgb_project)
      : velodyne_to_rgb(velodyne_to_rgb),
        rgb_project(rgb_project),
        folder_(folder_),
        fname_format_(fname_format_),
        data_buffer_(new float[kBufferSize]),
        latest_frame_(nullptr),
        latest_point_count_(0)
  {}

  virtual ~Velodyne() {
    delete data_buffer_;
  }

  /// \brief Returns an Nx4 **row-major** Eigen matrix containing the Velodyne readings from the
  ///        specified frame of the current dataset.
  LidarReadings ReadFrame(int frame_idx);

  /// \brief Returns an Nx4 **row-major** Eigen matrix containing the Velodyne readings from the
  ///        latest read frame.
  LidarReadings GetLatestFrame();
};

}
}

#endif //DYNSLAM_VELODYNE_H

