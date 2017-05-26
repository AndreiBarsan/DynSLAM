

#include "Utils/BoundingBox.h"
#include "InstanceTracker.h"
#include "Track.h"

#include <iomanip>
#include <sstream>

namespace InstRecLib {
	namespace Reconstruction {
		using namespace std;
		using namespace InstRecLib::Segmentation;
		using namespace InstRecLib::Utils;

		float Track::ScoreMatch(const TrackFrame &new_frame) const {
			// TODO(andrei): Use fine mask, not just bounding box.
			// TODO(andrei): Ensure this is modular enough to allow many different matching strategies.
			// TODO-LOW(andrei): Take time into account---if I overlap perfectly but with a very old
			// track, the score should probably be discounted.

			assert(!this->frames_.empty() && "A track with no frames cannot exist.");
			const TrackFrame &latest_frame = this->frames_[ this->frames_.size() - 1];

			// We don't want to accidentally add multiple segments from the same frame to the same track.
			// This is not 100% elegant, but it works.
			if (new_frame.frame_idx == this->GetEndTime()) {
				return 0.0f;
			}

			const InstanceDetection& new_detection = new_frame.instance_view.GetInstanceDetection();
			const InstanceDetection& latest_detection = latest_frame.instance_view.GetInstanceDetection();

			// We don't want to associate segments from different classes.
			// TODO(andrei): Sometimes the segmentation pipeline may flicker between, e.g., ``car'' and
			// ``truck'' so we may want a more complex reasoning system here in the future.
			if (new_detection.class_id != latest_detection.class_id) {
				return 0.0f;
			}

			const BoundingBox& new_bbox = new_detection.GetBoundingBox();
			const BoundingBox& last_bbox = latest_detection.GetBoundingBox();
			// Using the max makes sure we prefer matching to previous tracks with larger areas if the
			// new detection is also large. This increases robustness to small spurious detections,
			// preventing them from latching onto good tracks.
			int max_area = std::max(new_bbox.GetArea(), last_bbox.GetArea());
			int overlap_area = last_bbox.IntersectWith(new_bbox).GetArea();

			// If the overlap completely covers one of the frames, then it's considered perfect.
			// Otherwise,	frames which only partially intersect get smaller scores, and frames which don't
			// intersect at all get a score of 0.0.
			float area_score = static_cast<float>(overlap_area) / max_area;

			// Modulate the score by the detection probability. If we see a good overlap but it's a dodgy
			// detection, we may not want to add it to the track. For instance, when using MNC for
			// segmentation, it may sometimes detect both part of a car and the entire car as separate
			// instances. Luckily, in these situations, the proper full detection gets a score near 1.0,
			// while the partial one is around 0.5-0.6. We'd prefer to fuse in the entire car, so we
			// take the probability into account. Similarly, in a future frame, we prefer adding the
			// new data to the track with the most confident detections.
			float score = area_score * new_detection.class_probability * latest_detection.class_probability;

			return score;
		}

		string Track::GetAsciiArt() const {
			stringstream out;
			out << "Object #" << setw(4) << id_ << " [";
			int idx = 0;
			for(const TrackFrame& frame : frames_) {
				while(idx < frame.frame_idx) {
					out << "   ";
					++idx;
				}
				out << setw(3) << frame.frame_idx;
				++idx;
			}
			out << "]";

			return out.str();
		}
	}
}