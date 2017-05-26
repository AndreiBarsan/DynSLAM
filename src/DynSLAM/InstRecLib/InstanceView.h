

#ifndef INFINITAM_INSTANCEVIEW_H
#define INFINITAM_INSTANCEVIEW_H

#include "../../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "InstanceSegmentationResult.h"

// TODO remove
#include <iostream>

namespace InstRecLib {
	namespace Reconstruction {

		/// \brief Like ITMView, but associated with a particular object instance.
		class InstanceView {
		public:
			InstanceView(const Segmentation::InstanceDetection &instance_detection_,
									 const std::shared_ptr<ITMLib::Objects::ITMView> &view_)
					: instance_detection_(instance_detection_), view_(view_) { }

			virtual ~InstanceView() { }

			ITMLib::Objects::ITMView& GetView() {
				return *(view_.get());
			}

			const ITMLib::Objects::ITMView& GetView() const {
				return *(view_.get());
			}

			InstRecLib::Segmentation::InstanceDetection& GetInstanceDetection() {
				return instance_detection_;
			}

			const InstRecLib::Segmentation::InstanceDetection& GetInstanceDetection() const {
				return instance_detection_;
			}

		private:
			/// \brief Holds label, mask, and bounding box information.
			InstRecLib::Segmentation::InstanceDetection instance_detection_;

			/// \brief Holds the depth and RGB information about the object.
			std::shared_ptr<ITMLib::Objects::ITMView> view_;
		};

	}
}


#endif //INFINITAM_INSTANCEVIEW_H
