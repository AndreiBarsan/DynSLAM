#ifndef INSTRECLIB_INSTANCERECONSTRUCTOR_H
#define INSTRECLIB_INSTANCERECONSTRUCTOR_H

#include <Eigen/StdVector>

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

// TODO-LOW(andrei): Extract an interface from this class once its functionality is defined better,
// since the libviso-based implementation is just one possibility.
/// \brief Pipeline component responsible for reconstructing the individual object instances.
class InstanceReconstructor {
 public:
  static const vector<string> kClassesToReconstructVoc2012;
  static const vector<string> kPossiblyDynamicClassesVoc2012;

  static const std::vector<Eigen::Vector4i, Eigen::aligned_allocator<Eigen::Vector4i>> kMatplotlib2Palette;

  static bool ShouldReconstruct(const std::string &class_name) {
    return (find(
        kClassesToReconstructVoc2012.begin(),
        kClassesToReconstructVoc2012.end(),
        class_name) != kClassesToReconstructVoc2012.end());
  }

  static bool IsPossiblyDynamic(const std::string &class_name) {
    return (find(
        kPossiblyDynamicClassesVoc2012.cbegin(),
        kPossiblyDynamicClassesVoc2012.cend(),
        class_name) != kPossiblyDynamicClassesVoc2012.cend());
  }

  explicit InstanceReconstructor(InfiniTamDriver *driver, bool use_decay, bool enable_direct_refinement)
      : instance_tracker_(new InstanceTracker()),
        frame_idx_(0),
        driver_(driver),
        use_decay_(use_decay),
        enable_direct_refinement_(enable_direct_refinement),
        instance_depth_buffer_(driver->GetImageSize(), true, true),
        instance_color_buffer_(driver->GetImageSize(), true, true)
  {}

  /// \brief Uses the segmentation result to remove dynamic objects from the main view and save
  ///        them to separate buffers, which are then used for individual object reconstruction.
  ///
  /// This is the core of the instance reconstruction engine.
  ///
  /// \param dyn_slam The owner of this component.
  /// \param main_view The original InfiniTAM view of the scene. Gets mutated!
  /// \param segmentation_result The output of the view's semantic segmentation.
  /// \param scene_flow The sparse scene flow for the entire input.
  /// \param ssf_provider (Hacky) Sparse scene flow provider which can also be used for computing
  ///                     an object's relative pose between two frames, based on its associated
  ///                     scene flow vectors.
  /// \param always_separate Whether to always separately reconstruct car models, even if they're
  ///                        static (more expensive, but also more robust to cars changing their
  ///                        state from static to dynamic.
  void ProcessFrame(
      const dynslam::DynSlam* dyn_slam,
      ITMLib::Objects::ITMView *main_view,
      const segmentation::InstanceSegmentationResult &segmentation_result,
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
  );

  void ForceObjectCleanup(int object_id);

  void SaveObjectToMesh(int object_id, const string &fpath);

  bool IsDirectRefinementEnabled() const {
    return enable_direct_refinement_;
  }

  /// \brief Renders all available object depthmaps onto the given depthmap, obeying depth ordering.
  /// TODO(andrei): Improve performance.
  void CompositeInstanceDepthMaps(ITMFloatImage *out, const pangolin::OpenGlMatrix &model_view);

  /// \brief Adds instance color and depth onto the indicated buffers, using 'out_depth' as a
  ///        software Z-buffer.
  void CompositeInstances(ITMUChar4Image *out_color,
                          ITMFloatImage *out_depth,
                          dynslam::PreviewType preview_type,
                          const pangolin::OpenGlMatrix &model_view);

  /// Hacky method for checking whether there's an object getting reconstructed at the given coords.
  const Track& GetTrackAtPoint(int x, int y) const {
    return instance_tracker_->GetTrackAtPoint(x, y, frame_idx_);
  }

  /// Only for 'dynslam::eval' use.
  int GetFrameIdx_Evaluation() const {
    return frame_idx_;
  }

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

  /// \brief Experimental relative pose refinement using ITM's built-in trackers.
  /// Doesn't really work as of July 23, 2017.
  bool enable_itm_refinement_ = false;

  /// \brief Alternative approach based on semidense direct image alignment.
  bool enable_direct_refinement_;

  /// \brief Used for rendering instance-specific depth and color maps.
  ITMFloatImage instance_depth_buffer_;
  ITMUChar4Image instance_color_buffer_;

  /// \brief Updates the reconstruction associated with the object tracks, where applicable.
  void ProcessReconstructions(bool always_separate);

  /// \brief Fuses the frame with the specified ID into the track's 3D reconstruction.
  void FuseFrame(Track &track, size_t frame_idx) const;

  /// \brief Processes the latest frame of the given track, copying it to the appropriate
  ///        instance-specific frame if necessary.
  void ProcessSilhouette(Track &track,
                         ITMLib::Objects::ITMView *main_view,
                         const Eigen::Vector2i &frame_size,
                         const SparseSceneFlow &scene_flow,
                         bool always_separate) const;

  /// \brief Estimates object motion for every track, and populates instance views.
  /// The instance view are populated with RGB, depth, and scene flow information based on the
  /// semantic segmentation result.
  void UpdateTracks(const dynslam::DynSlam *dyn_slam,
                    const SparseSceneFlow &scene_flow,
                    const SparseSFProvider &ssf_provider,
                    bool always_separate,
                    ITMLib::Objects::ITMView *main_view,
                    const Eigen::Vector2i &frame_size) const;

  /// \brief Allocates and InfiniTAM instance for reconstructing the given object and fuses all the
  ///        available frames in the track.
  void InitializeReconstruction(Track &track) const;

  /// \brief Masks the scene flow using the (smaller) conservative mask of the instance detection.
  void ExtractSceneFlow(
      const SparseSceneFlow &scene_flow,
      vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &out_instance_flow_vectors,
      const segmentation::InstanceDetection &detection,
      const Eigen::Vector2i &frame_size,
      bool check_sf_start = true
  );

  /// \brief Converts segmentation results into "InstanceView" objects with associated RGB, depth,
  ///        and scene flow data.
  vector<InstanceView, Eigen::aligned_allocator<InstanceView>> CreateInstanceViews(
      const segmentation::InstanceSegmentationResult &segmentation_result,
      ITMLib::Objects::ITMView *main_view,
      const SparseSceneFlow &scene_flow
  );
};

}  // namespace reconstruction
}  // namespace instreclib

#endif  // INSTRECLIB_INSTANCERECONSTRUCTOR_H
