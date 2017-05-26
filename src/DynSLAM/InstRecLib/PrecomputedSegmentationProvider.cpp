

#include "PrecomputedSegmentationProvider.h"
#include "InstanceSegmentationResult.h"

// TODO(andrei): Get rid of this dependency.
#include "../../InfiniTAM/InfiniTAM/Utils/FileUtils.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace InstRecLib {
	namespace Segmentation {

		using namespace std;
		using namespace InstRecLib::Utils;

		/// \brief Reads a numpy text dump of an object's 2D binary segmentation mask.
		///
		/// \param np_txt_in Input stream from the numpy text file containing the mask.
		/// \param width The mask width, which is computed from its bounding box.
		/// \param height The mask height, also computed from its bounding box.
		/// \return A row-major array containing the mask info. The caller takes ownership.
		///
		/// \note The numpy file is already organized in 2D (as many lines as rows, etc.), so the given
		/// width and height are simply used as an additional sanity check.
		uint8_t** ReadMask(std::istream &np_txt_in, const int width, const int height) {
			// TODO(andrei): Is this stupidly slow?
			// TODO(andrei): Move to general utils or to IO library.
			// TODO(andrei): We could probably YOLO and work with binary anyway.
			int lines_read = 0;
			string line_buf;
			uint8_t **mask = new uint8_t*[height];
			memset(mask, 0, sizeof(size_t) * height);

			while(getline(np_txt_in, line_buf)) {
				if (lines_read >= height) {
					stringstream error_ss;
					error_ss << "Image height mismatch. Went over the given limit of " << height << ".";
					throw runtime_error(error_ss.str());
				}

				mask[lines_read] = new uint8_t[width];
				istringstream line_ss(line_buf);
				int col = 0;
				while(! line_ss.eof()) {
					if (col >= width) {
						stringstream error_ss;
						error_ss << "Image width mismatch. Went over the given limit of " << width << ".";
						cerr << "Line was:" << endl << endl << line_buf << endl << endl << flush;
						throw runtime_error(error_ss.str());
					}

					// TODO(andrei): Remove this once you update your dumping code to directly dump bytes.
					double val;
					line_ss >> val;
					mask[lines_read][col] = static_cast<uint8_t>(val);
					col++;
				}

				lines_read++;
			}

			return mask;
		}

		vector<InstanceDetection> PrecomputedSegmentationProvider::ReadInstanceInfo(
				const std::string &base_img_fpath
		) {
			// Loop through all possible instance detection dumps for this frame.
			//
			// They are saved by the pre-segmentation tool as:
			//     '${base_img_fpath}.${instance_idx}.{result,mask}.txt'.
			//
			// The result file is one line with the format "[x1 y1 x2 y2 junk], probability, class". The
			// first part represents the bounding box of the detection.
			//
			// The mask file is a numpy text file containing the saved (boolean) mask created by the neural
			// network. Its size is exactly the size of the bounding box.

			int instance_idx = 0;
			vector<InstanceDetection> detections;
			while (true) {
				stringstream result_fpath;
				result_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx
				             << ".result.txt";

				stringstream mask_fpath;
				mask_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx
				           << ".mask.txt";

				ifstream result_in(result_fpath.str());
				ifstream mask_in(mask_fpath.str());
				if (!(result_in.is_open() && mask_in.is_open())) {
					// No more detections to process.
					break;
				}

				// Process the result metadata file
				string result;
				getline(result_in, result);

				BoundingBox bounding_box;
				float class_probability;
				int class_id;
				sscanf(result.c_str(), "[%d %d %d %d %*d], %f, %d",
				       &bounding_box.r.x0, &bounding_box.r.y0, &bounding_box.r.x1, &bounding_box.r.y1,
				       &class_probability, &class_id);

				// Process the mask file. The mask area covers the edges of the bounding box, too.
				uint8_t **mask_pixels = ReadMask(mask_in, bounding_box.GetWidth(), bounding_box.GetHeight());
				auto mask = make_shared<Mask>(bounding_box, mask_pixels);
				detections.emplace_back(class_probability, class_id, mask, this->dataset_used);

				instance_idx++;
			}

			return detections;
		}

		shared_ptr<InstanceSegmentationResult> PrecomputedSegmentationProvider::SegmentFrame(
      ITMUChar4Image *rgb
		) {
			stringstream img_ss;
			img_ss << this->segFolder_ << "/" << "cls_" << setfill('0') << setw(6) << this->frameIdx_
			       << ".png";
			const string img_fpath = img_ss.str();
			ReadImageFromFile(lastSegPreview_, img_fpath.c_str());

			stringstream meta_ss;
			meta_ss << this->segFolder_ << "/" << setfill('0') << setw(6) << this->frameIdx_ << ".png";
			vector<InstanceDetection> instance_detections = ReadInstanceInfo(meta_ss.str());

			// We read data off the disk, so we assume this is 0.
			long inference_time_ns = 0L;

			this->frameIdx_++;

			return make_shared<InstanceSegmentationResult>(
					dataset_used,
			    instance_detections,
			    inference_time_ns
			);
		}

		const ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() const {
			return this->lastSegPreview_;
		}

		ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() {
			return this->lastSegPreview_;
		}
	}
}
