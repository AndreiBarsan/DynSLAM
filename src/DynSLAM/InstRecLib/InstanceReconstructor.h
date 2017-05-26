

#ifndef INFINITAM_INSTANCERECONSTRUCTOR_H
#define INFINITAM_INSTANCERECONSTRUCTOR_H

#include <memory>

#include "ChunkManager.h"
#include "InstanceSegmentationResult.h"
#include "InstanceTracker.h"

namespace InstRecLib {
	namespace Reconstruction {

		/// \brief Pipeline component responsible for reconstructing the individual object instances.
		class InstanceReconstructor {

		private:
			std::shared_ptr<InstanceTracker> instance_tracker_;

			// TODO(andrei): Consider keeping track of this in centralized manner and not just in UIEngine.
			/// \brief The current input frame number.
			/// Useful for, e.g., keeping track of when we last saw a car, so we can better associate
			/// detections through time, and dump old-enough reconstructions to the disk.
			int frame_idx_;

		public:
			InstanceReconstructor() : instance_tracker_(new InstanceTracker()),
			                          frame_idx_(0) { }

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
		};
	}
}


#endif //INFINITAM_INSTANCERECONSTRUCTOR_H
