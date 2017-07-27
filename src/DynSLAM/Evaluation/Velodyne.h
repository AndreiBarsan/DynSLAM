
#ifndef DYNSLAM_VELODYNE_H
#define DYNSLAM_VELODYNE_H

#include <string>

#include <Eigen/Eigen>
#include <fstream>
#include "../Utils.h"

namespace dynslam {
namespace eval {

class Velodyne {
 public:
  using LidarReadings = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;

  /// \brief The size of the buffer into which we initially read the data.
  const size_t kBufferSize = 1000000;
  /// \brief Each Velodyne point has 4 entries: X, Y, Z, and reflectance.
  static const unsigned short kMeasurementsPerPoint = 4;

 private:
  const std::string folder_;
  const std::string fname_format_;
  float * const data_buffer_;
  float * latest_frame_;
  size_t latest_point_count_;

 public:
  Velodyne(const std::string &folder_, const std::string &fname_format_)
      : folder_(folder_),
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

