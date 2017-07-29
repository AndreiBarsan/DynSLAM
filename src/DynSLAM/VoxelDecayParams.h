
#ifndef DYNSLAM_VOXELDECAYPARAMS_H
#define DYNSLAM_VOXELDECAYPARAMS_H

namespace dynslam {

struct VoxelDecayParams {
  /// \brief Whether to enable voxel decay.
  bool enabled;
  /// \brief Voxels older than this are eligible for decay.
  int min_decay_age;
  /// \brief Voxels with a weight smaller than this are decayed, provided that they are old enough.
  int max_decay_weight;

  VoxelDecayParams(bool enabled, int min_decay_age, int max_decay_weight)
      : enabled(enabled), min_decay_age(min_decay_age), max_decay_weight(max_decay_weight) {}
};

}

#endif //DYNSLAM_VOXELDECAYPARAMS_H
