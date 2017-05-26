

#ifndef INFINITAM_SEGMENTATIONDATASET_H
#define INFINITAM_SEGMENTATIONDATASET_H

#include <map>
#include <string>
#include <vector>

namespace InstRecLib {
	namespace Segmentation {

		/// \brief Builds a reverse mapping from label names to their indices in the list.
		std::map<std::string, int> labels_to_id_map(const std::vector<std::string> labels);

		/// \brief Describes a segmentation dataset, e.g., Pascal VOC, in terms such as supported labels.
		/// Does not, at this time, handle anything beyond label management, such as IO.
		struct SegmentationDataset {
			const std::string name;
			const std::vector<std::string> labels;
			const std::map<std::string, int> label_to_id;

			SegmentationDataset(const std::string name, const std::vector<std::string> labels)
					: name(name), labels(labels), label_to_id(labels_to_id_map(labels)) {}
		};

		//region Dataset descriptions
		const std::vector<std::string> kPascalVoc2012Classes = {
			"INVALID",  // VOC 2012 class IDs are 1-based.
			"airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
	  	"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
		  "sheep", "sofa", "train", "tvmonitor"
		};

		const SegmentationDataset kPascalVoc2012("Pascal VOC 2012", kPascalVoc2012Classes);
	}
}


#endif //INFINITAM_SEGMENTATIONDATASET_H
