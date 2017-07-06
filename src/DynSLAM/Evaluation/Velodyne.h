
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
  Eigen::MatrixXf ReadFrame(int frame_idx) {
    using namespace std;
    string fpath_format = dynslam::utils::Format("%s/%s", folder_.c_str(), fname_format_.c_str());
    string fpath = dynslam::utils::Format(fpath_format, frame_idx);

    utils::Tic("Velodyne dump read");
    FILE *velo_in = fopen(fpath.c_str(), "rb");
    size_t read_floats = fread(data_buffer_, sizeof(float), kBufferSize, velo_in);

    size_t read_points = read_floats / kMeasurementsPerPoint;
    cout << "Read " << read_points << " velo points from " << fpath << "." << endl;
    fclose(velo_in);
    utils::Toc();

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

  /// \brief Renders the velodyne points for visual inspection.
  /// \param velodyne_points
  /// \param P (Left) camera pose matrix.
  /// \param Tr Transforms velodyne points into the 0th camera's frame.
  /// \note For fused visualization we need to use the depth render as a zbuffer when rendering
  /// LIDAR points, either in OpenGL, or manually by projecting LIDAR points and manually checking
  /// their resulting depth. But we don't need this visualization yet; So far, it's enough to render
  /// the LIDAR results for sanity, and then for every point in the cam frame look up the model
  /// depth and compare the two.
  void RenderFrame(const Eigen::MatrixXf &velodyne_points,
                   const Eigen::MatrixXf &P,
                   const Eigen::MatrixXf &Tr
  ) {
    // convert every velo point into 2D as: x_i = P * Tr * X_i

  }

 private:
  const std::string folder_;
  const std::string fname_format_;

  float *data_buffer_;
};

}
}

#endif //DYNSLAM_VELODYNE_H

