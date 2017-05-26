

#ifndef INFINITAM_SEGMENTATIONRESULT_H
#define INFINITAM_SEGMENTATIONRESULT_H

#include "SegmentationDataset.h"
#include "Utils/Mask.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace InstRecLib {
	namespace Segmentation {

		class InstanceSegmentationResult;

		// TODO(andrei): Consider storing an entire class-conditional distribution in each detection
		// object.

		/*
		 * TODO(andrei): Move these notes away from here.
		 *
		 * Yeah, it does make more sense to have a SegmentationAssociationProvider which takes in an
		 * InstanceSegmentationResult with its InstanceDetections, and outputs e.g., a list of
		 * (InstanceDetection, InstanceReconstruction) pairs, which can then be used for volum. Fusion.
		 *
		 * The Association Provider can be just an interface, and it could have multiple implementations
		 * under the hood, such as overlap-based (with configurable size/time/etc. thresholds),
		 * object recognition-based, hybrid, etc.
		 *
		 * Its job is to be able to associate objects detected in a frame with their appropriate 3D
		 * model, or to initialize new models as needed.
		 *
		 * Then, we could cut out the instance info from the current frame, and give these cut-out bits
		 * to the instance rec engine, in their equivalent ProcessFrame method. The only extra necessary
		 * thing is the approximate relative pose of the new frame, perhaps in the viewer's frame. Then
		 * the instance rec would be able to initialize its alignment system, e.g., ICP, accordingly,
		 * based on this initial estimate. We can get this estimate from scene flow.
		 *
		 * Such an instance reconstructor may also need to keep track of the instance's motion (either
		 * in the global frame or the car's frame; we'll see which works best---probably the global
		 * frame, since it seems neater), as well as other things, such as (maybe) the motion history,
		 * allowing us to render nice tracks in the 3D map, etc.
		 */

		/// \brief Describes a single object instance detected semantically in an input frame.
		/// This is a component of InstanceSegmentationResult.
		class InstanceDetection {
		public:
			/// This detection is the highest-likelihood class from a proposal region.
			/// This is its probability. It's usually pretty high (>0.9), but can
			/// still be useful for various sanity checks.
			float class_probability;

			/// Class identifier. Depends on the class labels used by the segmentation pipeline,
			/// specified in the associated `SegmentationDataset`.
			int class_id;

			/// \brief 2D mask of this detection in its source image frame.
			std::shared_ptr<InstRecLib::Utils::Mask> mask;

			/// \brief The dataset associated with this instance's detection. Contains information such
			/// as mappings from class IDs to class names.
			const SegmentationDataset* segmentation_dataset;

			std::string GetClassName() const;

			InstRecLib::Utils::BoundingBox& GetBoundingBox() {
				return mask->GetBoundingBox();
			}

			const InstRecLib::Utils::BoundingBox& GetBoundingBox() const {
				return mask->GetBoundingBox();
			}

			/// \brief Initializes the detection with bounding box, class, and mask information.
			/// \note This object takes ownership of `mask`.
			InstanceDetection(float class_probability, int class_id,
												std::shared_ptr<InstRecLib::Utils::Mask> mask,
												const SegmentationDataset* segmentation_dataset)
					: class_probability(class_probability),
						class_id(class_id),
						mask(mask),
						segmentation_dataset(segmentation_dataset) { }

			virtual ~InstanceDetection() { }
		};

		/// \brief Supports pretty-printing segmentation result objects.
		std::ostream& operator<<(std::ostream& out, const InstanceDetection& detection);

		/// \brief The result of performing instance-aware semantic segmentation on an input frame.
		struct InstanceSegmentationResult {

			/// \brief Specifies the dataset metadata, such as the labels, which are used by the segmentation.
			const SegmentationDataset *segmentation_dataset;

			/// \brief All object instances detected in a frame.
			std::vector<InstanceDetection> instance_detections;

			/// \brief The total time it took to produce this result.
			long inference_time_ns;

			InstanceSegmentationResult(const SegmentationDataset *segmentation_dataset,
			                           const std::vector<InstanceDetection> &instance_detections,
			                           long inference_time_ns
			) : segmentation_dataset(segmentation_dataset),
			    instance_detections(instance_detections),
			    inference_time_ns(inference_time_ns)
			{ }
		};
	}
}


#endif //INFINITAM_SEGMENTATIONRESULT_H
