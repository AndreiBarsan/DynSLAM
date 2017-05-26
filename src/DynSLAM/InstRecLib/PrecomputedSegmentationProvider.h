

#ifndef INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
#define INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H

#include <memory>

#include "SegmentationProvider.h"
#include "InstanceSegmentationResult.h"

namespace InstRecLib {
	namespace Segmentation {

		/// \brief Reads pre-existing frame segmentations from the disk, instead of computing them on-the-fly.
		class PrecomputedSegmentationProvider : public SegmentationProvider {

		private:
			std::string segFolder_;
			int frameIdx_ = 0;
			ITMUChar4Image *lastSegPreview_;
			const SegmentationDataset* dataset_used;

		protected:
			std::vector<InstanceDetection> ReadInstanceInfo(const std::string &base_img_fpath);

		public:

			PrecomputedSegmentationProvider(const std::string &segFolder)
					: segFolder_(segFolder), dataset_used(&kPascalVoc2012) {
				printf("Initializing pre-computed segmentation provider.\n");

				lastSegPreview_ = new ITMUChar4Image(true, false);
			}

			~PrecomputedSegmentationProvider() override {
				delete lastSegPreview_;
			}

			std::shared_ptr<InstanceSegmentationResult> SegmentFrame(ITMUChar4Image *view) override;

			const ITMUChar4Image *GetSegResult() const override;

			ITMUChar4Image *GetSegResult() override;
		};
	}
}


#endif //INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
