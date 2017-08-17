
#ifndef DYNSLAM_VELODYNE_H
#define DYNSLAM_VELODYNE_H

#include <string>

#include <Eigen/Eigen>
#include <fstream>
#include "../Defines.h"
#include "../Utils.h"

namespace dynslam {
namespace eval {

/// \brief Reads and manages *corrected* Velodyne point clouds.
/// \note From the documentation of the KITTI-odometry dataset: Note that the velodyne scanner takes
/// depth measurements/continuously while rotating around its vertical axis (in contrast to the
/// cameras, which are triggered at a certain point in time). This effect has been eliminated from
/// this postprocessed data by compensating for the egomotion!! Note that this is in contrast to the
/// raw data.
class VelodyneIO {
 public:
  using LidarReadings = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;

  /// \brief Each Velodyne reading has 4 components: X, Y, Z, and reflectance.
  static const unsigned short kMeasurementsPerPoint = 4;

  /// \brief The size of the buffer into which we initially read the data.
  static const size_t kBufferSize = 1000000;

  /// \brief 4x4 matrix which transforms 3D homogeneous coordinates from the Velodyne LIDAR's
  ///        coordinate frame to the (typically left gray) camera's coordinate frame.
//  const Eigen::Matrix4d velodyne_to_cam_;

 private:
  const std::string folder_;
  const std::string fname_format_;
  float * const data_buffer_;
  float * latest_frame_;
  size_t latest_point_count_;

 public:
  SUPPORT_EIGEN_FIELDS;

  VelodyneIO(const std::string &folder, const std::string &fname_format)
      : folder_(folder),
        fname_format_(fname_format),
        data_buffer_(new float[kBufferSize]),
        latest_frame_(nullptr),
        latest_point_count_(0)
  {}

  virtual ~VelodyneIO() {
    delete data_buffer_;
  }

  /// \brief Checks if Velodyne data exists for the specified frame. Some frames do not have it
  ///        available.
  bool FrameAvailable(int frame_idx);

  /// \brief Returns an Nx4 **row-major** Eigen matrix containing the Velodyne readings from the
  ///        specified frame of the current dataset.
  LidarReadings ReadFrame(int frame_idx);

  /// \brief Returns an Nx4 **row-major** Eigen matrix containing the Velodyne readings from the
  ///        latest read frame.
  LidarReadings GetLatestFrame();

  bool HasLatestFrame() const {
    return nullptr != latest_frame_;
  }

 private:
  std::string GetVeloFpath(int frame_idx) const {
    std::string fpath_format = utils::Format("%s/%s", folder_.c_str(), fname_format_.c_str());
    return utils::Format(fpath_format, frame_idx);
  }
};

}
}

#endif //DYNSLAM_VELODYNE_H

