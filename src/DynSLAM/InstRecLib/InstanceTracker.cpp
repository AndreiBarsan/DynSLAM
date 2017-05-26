

#include "InstanceTracker.h"

#include <cassert>

namespace InstRecLib {
	namespace Reconstruction {

		using namespace std;
		using namespace InstRecLib::Segmentation;

		void InstanceTracker::ProcessInstanceViews(int frame_idx, const vector<InstanceView>& new_views) {
			cout << "Frame [" << frame_idx << "]. Processing " << new_views.size()
			     << " new detections." << endl;

			list<TrackFrame> new_track_frames;
			for(const InstanceView& view : new_views) {
				new_track_frames.emplace_back(frame_idx, view);
			}

			// 1. Find a matching track.
			this->AssignToTracks(new_track_frames);

			// 2. For leftover detections, put them into new, single-frame, tracks.
			cout << new_track_frames.size() << " new unassigned frames." << endl;
			for (const TrackFrame &track_frame : new_track_frames) {
				cout << "New track created. ID = " << track_count_ << "." << endl;
				Track new_track(track_count_++);
				new_track.AddFrame(track_frame);
				this->active_tracks_.push_back(new_track);
			}

			cout << "We now have " << this->active_tracks_.size() << " active track(s)." << endl;

			// 3. Iterate through tracks, find ``expired'' ones, and discard them.
			this->PruneTracks(frame_idx);
		}

		void InstanceTracker::PruneTracks(int current_frame_idx) {
			auto it = active_tracks_.begin();
			while(it != active_tracks_.end()) {
				int last_active = it->GetEndTime();
				int frame_delta = current_frame_idx - last_active;

				if (frame_delta > inactive_frame_threshold_) {
//					cout << "Erasing track of size " << it->GetSize() << "." << endl;
					it = active_tracks_.erase(it);
//					cout << "Done." << endl;
				}
				else {
					++it;
				}
			}
		}

		std::pair<Track *, float> InstanceTracker::FindBestTrack(const TrackFrame &track_frame) {
			if (active_tracks_.empty()) {
				return kNoBestTrack;
			}

			float best_score = -1.0f;
			Track *best_track = nullptr;

			for (Track& track : active_tracks_) {
				float score = track.ScoreMatch(track_frame);
				if (score > best_score) {
					best_score = score;
					best_track = &track;
				}
			}

			assert(best_score >= 0.0f);
			assert(best_track != nullptr);
			return std::pair<Track*, float>(best_track, best_score);
		}

		void InstanceTracker::AssignToTracks(std::list<TrackFrame> &new_detections) {
			auto it = new_detections.begin();
			while(it != new_detections.end()) {
				pair<Track*, float> match = FindBestTrack(*it);
				Track* track = match.first;
				float score = match.second;

				if (score > kTrackScoreThreshold) {
					cout << "Found a match based on overlap with score " << score << "." << endl;
					cout << "Adding it to track #" << track->GetId() << " of length " << track->GetSize() << "." << endl;

					track->AddFrame(*it);
					it = new_detections.erase(it);
				}
				else {
//					cout << "Best score was: " << score << ", below the threshold. Will create new track."
//					     << endl;
					++it;
				}
			}
		}
	}
}

