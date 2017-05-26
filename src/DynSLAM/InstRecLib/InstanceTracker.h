

#ifndef INFINITAM_INSTANCETRACKER_H
#define INFINITAM_INSTANCETRACKER_H


#include <list>
#include <vector>
#include <cassert>

#include "InstanceSegmentationResult.h"
#include "Track.h"
#include "InstanceView.h"

// TODO(andrei): Consider splitting this file up into multiple class files.
namespace InstRecLib {
	namespace Reconstruction {

		/// \brief Minimum overlap score required to add a new frame to an existing feature track.
		/// Between 0.0 and 1.0.
		const float kTrackScoreThreshold = 0.15f;

		// TODO(andrei): Adapt this to allow bigger gaps in longer track, maybe.
		/// \brief Default age of the last frame in an object track after which we discard it.
		/// The smaller this is, the less memory the system uses, but the likelier it is to fragment
		/// object reconstructions into multiple volumes.
		const int kDefaultInactiveFrameThreshold = 3;

		/// \brief Tracks instances over time by associating multiple isolated detections.
		// TODO(andrei): Once complete, refactor this into an interface + MaxOverlapTracker.
		class InstanceTracker {
		private:
			/// \brief The tracks of the objects being currently tracked.
			std::vector<Track> active_tracks_;

			/// \brief The maximum age of the latest frame in a track, before it is discarded.
			/// The higher this is, the more tracks are held in memory.
			int inactive_frame_threshold_;

			/// \brief The total number of tracks seen, including both active and pruned tracks.
			int track_count_;

		protected:
			static constexpr std::pair<Track*, float> kNoBestTrack = std::pair<Track*, float>(nullptr, 0.0f);

			/// \brief Finds the most likely track for the given frame, if it exists.
			/// \return The best matching track, and the [0-1] match quality score. If no tracks are
			/// available, kNoTrack is returned.
			std::pair<Track*, float> FindBestTrack(const TrackFrame& track_frame);

			/// \brief Assign the detections to the best matching tracks.
			/// \note Mutates the `new_detections` input list, removing the matched detections.
			void AssignToTracks(std::list<TrackFrame>& new_detections);

			/// \brief Removes tracks which have not been active in the past k frames.
			void PruneTracks(int current_frame_idx);

		public:
			InstanceTracker() : active_tracks_(std::vector<Track>()),
			                    inactive_frame_threshold_(kDefaultInactiveFrameThreshold),
													track_count_(0) { }

			/// \brief Associates the new detections with existing tracks, or creates new ones.
			/// \param new_detections The instances detected in the current frame.
			void ProcessInstanceViews(int frame_idx, const std::vector<InstanceView>& new_detections);

			std::vector<Track>& GetTracks() {
				return active_tracks_;
			}

			const std::vector<Track>& GetTracks() const {
				return active_tracks_;
			}

			/// \see track_count_
			int GetTotalTrackCount() const {
				return track_count_;
			}

			int GetActiveTrackCount() const {
				return static_cast<int>(active_tracks_.size());
			}
		};

	}
}



#endif //INFINITAM_INSTANCETRACKER_H
