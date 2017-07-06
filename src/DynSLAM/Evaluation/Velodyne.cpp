
#include "Velodyne.h"

namespace dynslam {
namespace eval {

Eigen::MatrixXf Velodyne::ReadFrame(int frame_idx) {
  using namespace std;
  string fpath_format = dynslam::utils::Format("%s/%s", folder_.c_str(), fname_format_.c_str());
  string fpath = dynslam::utils::Format(fpath_format, frame_idx);

//    utils::Tic("Velodyne dump read");
  FILE *velo_in = fopen(fpath.c_str(), "rb");
  size_t read_floats = fread(data_buffer_, sizeof(float), kBufferSize, velo_in);

  size_t read_points = read_floats / kMeasurementsPerPoint;
//  cout << "Read " << read_points << " velo points from " << fpath << "." << endl;
    fclose(velo_in);
//    utils::Toc();

  // Ensure we convert from row-major to column-major data, since Eigen stores matrices in
  // column-major format. Performance-wise, row-major would be faster, but we'd rather stay with
  // the safe defaults for the evaluation code.
  using RowMajor = Eigen::Stride<1, 4>;
  Eigen::MatrixXf points = Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned, RowMajor>(
      data_buffer_, read_points, 4);

  // From the incremental dense semantic stereo ITM paper:
  /*
   * In order to evaluate accuracy, we follow the approach of Sengupta et al. [4], who measure
   * the number of pixels whose distance (in terms of depth) from the ground truth (in our case
   * the Velodyne data) after projection to the image plane is less than a fixed threshold.
   */
  // => TODO(andrei): Project LIDAR points into 2D and display.
  // => TODO(andrei): Project map (or use current live raycast?) and compare to LIDAR.
  // => TODO(andrei): Plot error as fn of delta, the allowed error. (see Sengupta/Torr paper Fig 10).

  return points;
}

}
}
