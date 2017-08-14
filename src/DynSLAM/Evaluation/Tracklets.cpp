#include "Tracklets.h"

#include <iostream>

namespace dynslam {
namespace eval {

using namespace std;

TrackType GetTrackType(const std::string &track_type_name) {
//  std::cout << "Name2type for: [" << track_type_name << "]." << std::endl;
  return kNameToType.at(track_type_name);
}

std::string GetTrackTypeName(TrackType type) {
  // Dirty way
  for (auto el : kNameToType) {
    if(el.second == type) {
      return el.first;
    }
  }
  throw std::runtime_error("Unknown track type.");
}

std::string GetOcclusionLevelName(OcclusionLevel occlusion_level) {
  return kOcclusionLevelNames.at(occlusion_level);
}

OcclusionLevel GetOcclusionLevel(int occlusion_level_id) {
  // We could do a direct cast, but let's be a little more generic and paranoid!
  return kIdToOcclusionLevel.at(occlusion_level_id);
}

std::istream &operator>>(std::istream &input, TrackType &track_type) {
  std::string name;
  input >> name;
  track_type = GetTrackType(name);
  return input;
}

std::istream &operator>>(std::istream &input, OcclusionLevel &occlusion_level) {
  int level_id;
  input >> level_id;
  occlusion_level = GetOcclusionLevel(level_id);
  return input;
}

std::istream &operator>>(std::istream &input, TrackletFrame &output) {
  int frame;
  if (input >> frame) {
    output.frame = frame;
    input >> output.track_id >> output.type >> output.truncated
          >> output.occlusion_level >> output.alpha
          >> output.bbox_2d[0] >> output.bbox_2d[1] >> output.bbox_2d[2] >> output.bbox_2d[3]
          >> output.dimensions_m(0) >> output.dimensions_m(1) >> output.dimensions_m(2)
          >> output.location_cam_m(0) >> output.location_cam_m(1) >> output.location_cam_m(2)
          >> output.rotation_y;
  }
  return input;
}

std::ostream &operator<<(std::ostream &out, const TrackletFrame &tf) {
  out << "TrackletFrame["
      << "frame = " << tf.frame << ", type = " << GetTrackTypeName(tf.type)
      << ", truncated = " << tf.truncated << ", occlusion_level = "
      << GetOcclusionLevelName(tf.occlusion_level) << ", alpha = " << tf.alpha << ", bbox = ["
      << tf.bbox_2d[0] << ", " << tf.bbox_2d[1] << ", " << tf.bbox_2d[2] << ", " << tf.bbox_2d[3]
      << "], dimensions_m = " << tf.dimensions_m(0) << " x " << tf.dimensions_m(1) << " x "
      << tf.dimensions_m(2) << ", location = " << tf.location_cam_m(0) << ", " << tf.location_cam_m(1)
      << ", " << tf.location_cam_m(2) << ", rotation_y = " << tf.rotation_y << "]";
  return out;
}

std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>
ReadTracklets(const std::string &fpath, bool cars_only)
{
  std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>> sequence_tracklets;

  std::ifstream input(fpath);
  if (! input.is_open()) {
    throw std::runtime_error(dynslam::utils::Format(
        "Could not read tracklet ground truth from [%s].", fpath.c_str()));
  }

  TrackletFrame result;
  while (input >> result) {
//    cout << "Read: " << result << endl;
    if (!cars_only || result.type == kCar) {
      sequence_tracklets.push_back(result);
    }
  }

  return sequence_tracklets;
}

std::map<int, std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>>
ReadGroupedTracklets(const std::string &fpath, bool cars_only)
{
  std::map<int, std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>> result;
  auto all = ReadTracklets(fpath, cars_only);

  for(const TrackletFrame &tf : all) {
    result[tf.frame].push_back(tf);
  }

  return result;
}

}
}
