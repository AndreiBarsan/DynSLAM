

#ifndef INFINITAM_INSTANCERECONSTRUCTOR_H
#define INFINITAM_INSTANCERECONSTRUCTOR_H

#include <map>
#include <memory>

#include "ChunkManager.h"
#include "InstanceSegmentationResult.h"
#include "InstanceTracker.h"

#include "../InfiniTamDriver.h"


namespace InstRecLib { namespace Reconstruction {

using namespace dynslam::drivers;

/// \brief Pipeline component responsible for reconstructing the individual object instances.
class InstanceReconstructor {
public:
  InstanceReconstructor(InfiniTamDriver *driver)
    : instance_tracker_(new InstanceTracker()), frame_idx_(0), driver(driver) { }

  /// \brief Uses the segmentation result to remove dynamic objects from the main view and save
  /// them to separate buffers, which are then used for individual object reconstruction.
  ///
  /// This is the ``meat'' of the reconstruction engine.
  ///
  /// \param main_view The original InfiniTAM view of the scene. Gets mutated!
  /// \param segmentation_result The output of the view's semantic segmentation.
  void ProcessFrame(
      ITMLib::Objects::ITMView* main_view,
      const Segmentation::InstanceSegmentationResult& segmentation_result
  );

  const InstanceTracker& GetInstanceTracker() const {
    return *instance_tracker_;
  }

  InstanceTracker& GetInstanceTracker() {
    return *instance_tracker_;
  }

  int GetActiveTrackCount() const {
    return instance_tracker_->GetActiveTrackCount();
  }

  /// \brief Returns a snapshot of one of the stored instance segments, if available.
  /// This method is primarily designed for visualization purposes.
  ITMUChar4Image *GetInstancePreviewRGB(size_t track_idx);

  ITMFloatImage *GetInstancePreviewDepth(size_t track_idx);

  void GetInstanceRaycastPreview(ITMUChar4Image *out) {
    // hacky, for very early preview
    if (id_to_reconstruction_.cend() != id_to_reconstruction_.find(0)) {
      id_to_reconstruction_.at(0)->GetImage(out, ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);
    }
  }

private:
  std::shared_ptr<InstanceTracker> instance_tracker_;

  // TODO(andrei): Consider keeping track of this in centralized manner and not just in UIEngine.
  /// \brief The current input frame number.
  /// Useful for, e.g., keeping track of when we last saw a car, so we can better associate
  /// detections through time, and dump old-enough reconstructions to the disk.
  int frame_idx_;

  std::map<int, InfiniTamDriver*> id_to_reconstruction_;
  // A bit hacky, but used as a "template" when allocating new reconstructors for objects.
  InfiniTamDriver *driver;

  void ProcessReconstructions() {
    using namespace std;

    for (Track& track : instance_tracker_->GetTracks()) {
      // TODO(andrei): proper heuristic to determine which tracks are worth reconstructing, e.g.,
      // based on slice surface, length, gaps, etc.

      // Since this is very memory-hungry, we restrict creation to the very first thing we see
      if (track.GetId() == 0) {
        if (id_to_reconstruction_.find(track.GetId()) == id_to_reconstruction_.cend()) {
          // Update reconstruction
          cout << "Starting to reconstruct instance with ID: " << track.GetId() << endl;
          // TODO
          id_to_reconstruction_.emplace(make_pair(
            track.GetId(),
            new InfiniTamDriver(
              driver->GetSettings(),
              driver->GetView()->calib,
              driver->GetView()->rgb->noDims,
              driver->GetView()->rgb->noDims
            )
          ));
        }
        else {
          // TODO(andrei): Use some heuristic to avoid cases which are obviously crappy.
          cout << "Continuing to reconstruct instance with ID: " << track.GetId() << endl;
        }

        // This doesn't seem necessary, since we nab the instance view after the "global"
        // UpdateView which processes the depth.
//          id_to_reconstruction_[track.GetId()]->UpdateView(rgb, depth);
        // This replaces the "UpdateView" call.
        InfiniTamDriver *instance_driver = id_to_reconstruction_[track.GetId()];
        instance_driver->SetView(track.GetLastFrame().instance_view.GetView());
        // TODO(andrei): This seems like the place to shove in e.g., scene flow data.

        cout << endl << endl << "Start instance integration for #" << track.GetId() << endl;
        instance_driver->Track();
        instance_driver->Integrate();
        instance_driver->PrepareNextStep();

        cout << endl << endl << "Finished instance integration." << endl;
      }
      else {
        cout << "Won't create voxel volume for instance #" << track.GetId() << " in the current"
             << " experimental mode." << endl;
      }
    }
  }

};

} }


#endif //INFINITAM_INSTANCERECONSTRUCTOR_H
