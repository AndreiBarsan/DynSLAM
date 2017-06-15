

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

/// \brief Pipeline component responsible for reconstructing the individual object instances.
class InstanceReconstructor {
 public:
  InstanceReconstructor(InfiniTamDriver *driver)
      : instance_tracker_(new InstanceTracker()),
        frame_idx_(0),
        driver(driver) {}

  /// \brief Uses the segmentation result to remove dynamic objects from the main view and save
  /// them to separate buffers, which are then used for individual object reconstruction.
  ///
  /// This is the ``meat'' of the reconstruction engine.
  ///
  /// \param main_view The original InfiniTAM view of the scene. Gets mutated!
  /// \param segmentation_result The output of the view's semantic segmentation.
  void ProcessFrame(ITMLib::Objects::ITMView *main_view,
                    const segmentation::InstanceSegmentationResult &segmentation_result,
                    // TODO(andrei): Organize these args better.
                    const SparseSceneFlow &scene_flow,
                    const SparseSFProvider &ssf_provider);

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
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix()
  ) {
    if (instance_tracker_->HasTrack(object_idx)) {
      Track& track = instance_tracker_->GetTrack(object_idx);
      if (track.HasReconstruction()) {
        track.GetReconstruction()->GetImage(
            out,
            ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME,
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

  // TODO(andrei): Consider keeping track of this in centralized manner in DynSLAM.
  /// \brief The current input frame number.
  /// Useful for, e.g., keeping track of when we last saw a car, so we can better associate
  /// detections through time, and dump old-enough reconstructions to the disk.
  int frame_idx_;

  // TODO(andrei): Is this still necessary?
//  std::map<int, InfiniTamDriver *> id_to_reconstruction_;

  // A bit hacky, but used as a "template" when allocating new reconstructors for objects. This is a
  // pointer to the driver used for reconstructing the static scene.
  // TODO(andrei): Looks like a good place to use factories.
  InfiniTamDriver *driver;

  void ProcessReconstructions();
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCERECONSTRUCTOR_H
