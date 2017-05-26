#ifndef INFINITAM_RECONSTRUCTIONMANAGER_H
#define INFINITAM_RECONSTRUCTIONMANAGER_H

#include <map>
#include <memory>
#include <string>
#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"


namespace InstRecLib {
	namespace Reconstruction {

		/// \brief Keeps track of views of individual objects, as produced by the semantic segmentation.
		///
		/// After the segmentation, the dynamic objects' pixels are ``cut out'' of the original RGB and
		/// depth frames and put into individual buffers, which are then associated across frames and
		/// reconstructed, if enough information is available, or discarded otherwise (for instance, if
		/// a car is only seen for 2-3 frames, which is not enough for any useful 3D reconstruction).
		/// What is left of the original RGB and depth buffers after the dynamic objects are removed is
		/// considered the static environment.
		///
		/// TODO(andrei): Cool idea---directly support masking in the InfiniTAM library kernels, so that
		/// we don't need to split anything up. This would be nontrivial to implement, but could speed
		/// things up quite a bit.
		///
		/// Currently under development.
		class ChunkManager {
		private:
			// TODO(andrei): Consider just templating 'ChunkManager' on these things.
			/// \brief Uniquely identifies an object detected in a scene.
			using Fingerprint = std::string;
			using SceneView = ITMLib::Objects::ITMView;

			std::map<Fingerprint, std::shared_ptr<SceneView>> chunks_;

		public:
			ChunkManager() : chunks_() { }

			bool hasChunk(const Fingerprint &fingerprint) {
				return chunks_.find(fingerprint) != chunks_.end();
			}

			void createChunk(
					const Fingerprint &fingerprint,
					const ITMLib::Objects::ITMRGBDCalib* calibration,
					const Vector2i frame_size,
					bool use_gpu
			) {
				chunks_[fingerprint] = std::make_shared<SceneView>(
						calibration, frame_size, frame_size, use_gpu);
			}

			// TODO(andrei): Consider having logic for removing views for objects we haven't seen in k
			// frames. Should that logic even be here? Should we just send the commands to this component
			// from someplace else which would decide when a view is no longer needed?

			std::shared_ptr<SceneView> getChunk(const Fingerprint& fingerprint) {
				return chunks_[fingerprint];
			}
		};
	}
}

#endif //INFINITAM_RECONSTRUCTIONMANAGER_H
