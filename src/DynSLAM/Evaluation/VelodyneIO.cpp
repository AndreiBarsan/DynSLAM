
#include <iostream>
#include "VelodyneIO.h"

namespace dynslam {
namespace eval {

using namespace std;

bool VelodyneIO::FrameAvailable(int frame_idx) {
  return utils::FileExists(GetVeloFpath(frame_idx));
}

VelodyneIO::LidarReadings VelodyneIO::ReadFrame(int frame_idx) {
  string fpath = GetVeloFpath(frame_idx);
//  cout << "Reading LIDAR for frame " << frame_idx << endl;

//    utils::Tic("Velodyne dump read");
  FILE *velo_in = fopen(fpath.c_str(), "rb");
  if (nullptr == velo_in) {
    throw std::runtime_error(utils::Format("Could not read Velodyne data from file: [%s]", fpath.c_str()));
  }
  size_t read_floats = fread(data_buffer_, sizeof(float), kBufferSize, velo_in);
  fclose(velo_in);

  latest_point_count_ = read_floats / kMeasurementsPerPoint;
//    utils::Toc();

  latest_frame_ = (float *) malloc(sizeof(float) * read_floats);
  memcpy(latest_frame_, data_buffer_, sizeof(float) * read_floats);

  return GetLatestFrame();
}

VelodyneIO::LidarReadings VelodyneIO::GetLatestFrame() {
  assert(nullptr != latest_frame_ && "No frame read yet!");
  LidarReadings points = Eigen::Map<LidarReadings>(
      latest_frame_, latest_point_count_, kMeasurementsPerPoint);
  return points;
}

}
}
