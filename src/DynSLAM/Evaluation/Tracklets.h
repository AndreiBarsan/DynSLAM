#ifndef DYNSLAM_TRACKLETLOADER_H
#define DYNSLAM_TRACKLETLOADER_H

#include <Eigen/StdVector>

#include <string>
#include <map>
#include <Eigen/Core>
#include <fstream>

#include "../Defines.h"
#include "../Utils.h"

namespace dynslam {
namespace eval {

enum TrackType {
  kCar, kVan, kTruck, kPedestrian, kPersonSitting, kCyclist, kTram, kMisc, kDontCare
};

enum OcclusionLevel {
  kFullyVisible, kPartlyOccluded, kLargelyOccluded, kUnknown, kNotApplicable
};

const std::map<std::string, TrackType> kNameToType = {
    { "Car", kCar },
    { "Van", kVan },
    { "Truck", kTruck },
    { "Pedestrian", kPedestrian },
    { "Person_sitting", kPersonSitting },
    { "Cyclist", kCyclist },
    { "Tram", kTram },
    { "Misc", kMisc },
    { "DontCare", kDontCare }
};

const std::map<int, OcclusionLevel> kIdToOcclusionLevel = {
    { -1, kNotApplicable },
    { 0, kFullyVisible },
    { 1, kPartlyOccluded },
    { 2, kLargelyOccluded },
    { 3, kUnknown }
};

const std::map<OcclusionLevel, std::string> kOcclusionLevelNames = {
    { kNotApplicable, "Not applicable" },
    { kFullyVisible, "Fully visible" },
    { kPartlyOccluded, "Partly occluded" },
    { kLargelyOccluded, "Largely occluded" },
    { kUnknown, "Unknown occlusion" }
};

TrackType GetTrackType(const std::string &track_type_name);

std::string GetTrackTypeName(TrackType type);

std::string GetOcclusionLevelName(OcclusionLevel occlusion_level);

OcclusionLevel GetOcclusionLevel(int occlusion_level_id);

struct TrackletFrame {
  /// \brief The frame within the sequence where the object appears.
  int frame;
  /// \brief Unique track identifier for this frame.
  int track_id;
  TrackType type;
  int truncated;
  OcclusionLevel  occlusion_level;
  /// \brief Observation angle from the camera's perspective [-pi, pi].
  double alpha;
  /// Zero-based pixel coordinates: (left, top), (right, bottom).
  double bbox_2d[4];
  /// \brief Box dimensions, in meters.
  Eigen::Vector3d dimensions_m;
  /// \brief Box location, in camera coordinates, in meters.
  Eigen::Vector3d location_cam_m;
  /// \brief Rotation around Y (up) axis in camera coordinates [-pi, pi].
  double rotation_y;

  SUPPORT_EIGEN_FIELDS;
};

std::istream& operator>>(std::istream& input, TrackType &track_type);

std::istream& operator>>(std::istream& input, OcclusionLevel &occlusion_level);

std::istream& operator>>(std::istream& input, TrackletFrame &output);

std::ostream& operator<<(std::ostream& out, const TrackletFrame &tf);

/// \brief Reads KITTI tracking benchmark-style tracklet information.
std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>> ReadTracklets(
    const std::string &fpath,
    bool cars_only = true
);

std::map<int, std::vector<TrackletFrame, Eigen::aligned_allocator<TrackletFrame>>>
ReadGroupedTracklets(const std::string &fpath, bool cars_only = true);

}
}

#endif //DYNSLAM_TRACKLETLOADER_H
