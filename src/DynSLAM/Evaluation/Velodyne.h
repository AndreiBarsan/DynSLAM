
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
  const size_t kBufferSize = 1000000;
  /// \brief Each Velodyne point has 4 entries: X, Y, Z, and reflectance.
  const unsigned short kMeasurementsPerPoint = 4;

  Velodyne(const std::string &folder_, const std::string &fname_format_)
      : folder_(folder_), fname_format_(fname_format_), data_buffer_(new float[kBufferSize]) {}

  // TODO(andrei): Once the code is stable, convert to MatrixX4f.
  Eigen::MatrixXf ReadFrame(int frame_idx);

 private:
  const std::string folder_;
  const std::string fname_format_;

  float *data_buffer_;
};

}
}

#endif //DYNSLAM_VELODYNE_H

