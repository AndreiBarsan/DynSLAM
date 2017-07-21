

#ifndef INSTRECLIB_INSTANCERECONSTRUCTOR_H
#define INSTRECLIB_INSTANCERECONSTRUCTOR_H

#include <map>
#include <memory>

#include "InstanceSegmentationResult.h"
#include "InstanceTracker.h"

#include "../InfiniTamDriver.h"
#include "SparseSFProvider.h"

namespace dynslam {
  class DynSlam;
}

namespace instreclib {
namespace reconstruction {

using namespace dynslam::drivers;

// TODO(andrei): Extract an interface from this class once its functionality is defined better.
/// \brief Pipeline component responsible for reconstructing the individual object instances.
class InstanceReconstructor {
 public:

  static const vector<string> classes_to_reconstruct_voc2012;
  static const vector<string> possibly_dynamic_classes_voc2012;

  static bool ShouldReconstruct(const std::string &class_name) {
    return (find(
        classes_to_reconstruct_voc2012.begin(),
        classes_to_reconstruct_voc2012.end(),
        class_name) != classes_to_reconstruct_voc2012.end());
  }

  static bool IsPossiblyDynamic(const std::string &class_name) {
    return (find(
        possibly_dynamic_classes_voc2012.cbegin(),
        possibly_dynamic_classes_voc2012.cend(),
        class_name) != possibly_dynamic_classes_voc2012.cend());
  }

  InstanceReconstructor(InfiniTamDriver *driver, bool use_decay = true)
      : instance_tracker_(new InstanceTracker()),
        frame_idx_(0),
        driver_(driver),
        use_decay_(use_decay) {}

  /// \brief Uses the segmentation result to remove dynamic objects from the main view and save
  ///        them to separate buffers, which are then used for individual object reconstruction.
  ///
  /// This is the core of the reconstruction engine.
  ///
  /// \param dyn_slam The owner of this component.
  /// \param main_view The original InfiniTAM view of the scene. Gets mutated!
  /// \param segmentation_result The output of the view's semantic segmentation.
  /// \param always_separate Whether to always separately reconstruct car models, even if they're
  ///                        static (more expensive, but also more robust to cars changing their
  ///                        state from static to dynamic.
  void ProcessFrame(
      const dynslam::DynSlam* dyn_slam,
      ITMLib::Objects::ITMView *main_view,
      const segmentation::InstanceSegmentationResult &segmentation_result,
      // TODO(andrei): Organize these args better.
      const SparseSceneFlow &scene_flow,
      const SparseSFProvider &ssf_provider,
      bool always_separate
  );

  const InstanceTracker &GetInstanceTracker() const { return *instance_tracker_; }

  InstanceTracker &GetInstanceTracker() { return *instance_tracker_; }

  int GetActiveTrackCount() const { return instance_tracker_->GetActiveTrackCount(); }

  /// \brief Returns a snapshot of one of the stored instance segments, if available.
  /// This method is primarily designed for visualization purposes.
  ITMUChar4Image *GetInstancePreviewRGB(size_t track_idx);

  ITMFloatImage *GetInstancePreviewDepth(size_t track_idx);

  // TODO(andrei): Operate with Eigen matrices everywhere for consistency.
  void GetInstanceRaycastPreview(
      ITMUChar4Image *out,
      int object_idx,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix(),
      dynslam::PreviewType preview_type = dynslam::PreviewType::kNormal
  ) {
    if (instance_tracker_->HasTrack(object_idx)) {
      Track& track = instance_tracker_->GetTrack(object_idx);
      if (track.HasReconstruction()) {
        track.GetReconstruction()->GetImage(
            out,
            preview_type,
            model_view);

        return;
      }
    }

    // If we're not tracking the object, or if it has no reconstruction available, then there's
    // nothing to display.
    out->Clear();
  }

  void SaveObjectToMesh(int object_id, const string &fpath);

 private:
  std::shared_ptr<InstanceTracker> instance_tracker_;

  // TODO(andrei): Consider keeping track of this in centralized manner in DynSLAM, and having
  // components like this one hold a pointer to their daddy. 'Member XNA components?
  /// \brief The current input frame number.
  /// Useful for, e.g., keeping track of when we last saw a car, so we can better associate
  /// detections through time, and dump old-enough reconstructions to the disk.
  int frame_idx_;

  // A bit hacky, but used as a "template" when allocating new reconstructors for objects. This is a
  // pointer to the driver used for reconstructing the static scene.
  // TODO(andrei): Looks like a good place to use factories.
  InfiniTamDriver *driver_;

  /// \brief Whether to use voxel decay for regularizing the reconstructed objects.
  bool use_decay_;

  void ProcessReconstructions(bool always_separate);

  /// \brief Fuses the specified frame into the track's 3D reconstruction.
  void FuseFrame(Track &track, size_t frame_idx) const;

  /// \brief Processes the latest frame of the given track, copying it to the appropriate
  ///        instance-specific frame if necessary.
  void ProcessSilhouette(Track &track,
                         ITMLib::Objects::ITMView *main_view,
                         const Eigen::Vector2i &frame_size,
                         const SparseSceneFlow &scene_flow,
                         bool always_separate) const;

  void UpdateTracks(const SparseSceneFlow &scene_flow,
                    const SparseSFProvider &ssf_provider,
                    bool always_separate,
                    ITMLib::Objects::ITMView *main_view,
                    const Eigen::Matrix4f &egomotion,
                    const Eigen::Vector2i &frame_size) const;
  void InitializeReconstruction(Track &track) const;
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCERECONSTRUCTOR_H
