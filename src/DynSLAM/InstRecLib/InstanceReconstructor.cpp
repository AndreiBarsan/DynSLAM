

#include "InstanceReconstructor.h"
#include "InstanceView.h"

#include <vector>

namespace InstRecLib {
	namespace Reconstruction {
		using namespace std;
		using namespace InstRecLib::Segmentation;
		using namespace InstRecLib::Utils;
		using namespace ITMLib::Objects;

		// TODO(andrei): Implement this in CUDA. It should be easy.
		void processSilhouette_CPU(
				Vector4u *sourceRGB,
				float *sourceDepth,
				Vector4u *destRGB,
				float *destDepth,
				Vector2i sourceDims,
				const InstanceDetection &detection
		) {
			// Blanks out the detection's silhouette in the 'source' frames, and writes its pixels into
			// the output frames.
			// Initially, the dest frames will be the same size as the source ones, but this is wasteful
			// in terms of memory: we should use bbox+1-sized buffers in the future, since most
			// silhouettes are relatively small wrt the size of the whole frame.
			//
			// Moreover, we should be able to pass in several output buffer addresses and a list of
			// detections to the CUDA kernel, and do all the ``splitting up'' work in one kernel call. We
			// may need to add support for the adaptive-size output buffers, since otherwise writing to
			// e.g., 5-6 output buffers may end up using up way too much GPU memory.

			int frame_width = sourceDims[0];
			int frame_height = sourceDims[1];
			const BoundingBox& bbox = detection.GetBoundingBox();

			int box_width = bbox.GetWidth();
			int box_height = bbox.GetHeight();

			memset(destRGB, 0, frame_width * frame_height * sizeof(Vector4u));
			memset(destDepth, 0, frame_width * frame_height * sizeof(float));

			for(int row = 0; row < box_height; ++row) {
				for(int col = 0; col < box_width; ++col) {
					int frame_row = row + bbox.r.y0;
					int frame_col = col + bbox.r.x0;
					// TODO(andrei): Are the CPU-specific InfiniTAM functions doing this in a nicer way?

					int frame_idx = frame_row * frame_width + frame_col;

					int mask = detection.mask->GetMask()[row][col];
					if (mask == 1) {
						destRGB[frame_idx].r = sourceRGB[frame_idx].r;
						destRGB[frame_idx].g = sourceRGB[frame_idx].g;
						destRGB[frame_idx].b = sourceRGB[frame_idx].b;
						destRGB[frame_idx].a = sourceRGB[frame_idx].a;
						sourceRGB[frame_idx].r = 0;
						sourceRGB[frame_idx].g = 0;
						sourceRGB[frame_idx].b = 0;
						sourceRGB[frame_idx].a = 0;

						destDepth[frame_idx] = sourceDepth[frame_idx];
						sourceDepth[frame_idx] = 0.0f;
					}
				}
			}
		}

		void InstanceReconstructor::ProcessFrame(
				ITMLib::Objects::ITMView* main_view,
				const Segmentation::InstanceSegmentationResult& segmentation_result
		) {
			// TODO(andrei): Perform this slicing 100% on the GPU.
			main_view->rgb->UpdateHostFromDevice();
			main_view->depth->UpdateHostFromDevice();

			ORUtils::Vector4<unsigned char> *rgb_data_h = main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
			float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

			vector<InstanceView> new_instance_views;
			for(const InstanceDetection& instance_detection : segmentation_result.instance_detections) {
				// At this stage of the project, we only care about cars. In the future, this scheme could
				// be extended to also support other classes, as well as any unknown, but moving, objects.
				if (instance_detection.class_id == kPascalVoc2012.label_to_id.at("car")) {
					Vector2i frame_size = main_view->rgb->noDims;
					// bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
					// check this, since the field is private.
					bool use_gpu = true;
					auto view = make_shared<ITMView>(main_view->calib, frame_size, frame_size, use_gpu);
					auto rgb_segment_h = view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
					auto depth_segment_h = view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

					processSilhouette_CPU(rgb_data_h, depth_data_h, rgb_segment_h, depth_segment_h,
					                      main_view->rgb->noDims, instance_detection);

					new_instance_views.emplace_back(instance_detection, view);
				}
			}

			// Associate this frame's detection(s) with those from previous frames.
			this->instance_tracker_->ProcessInstanceViews(frame_idx_, new_instance_views);

			// TODO(andrei): Keep track (here or in instance_tracker_) of which tracks have their own
			// reconstructions. It's tempting to do all this in 'InstanceTracker', but it should only
			// be concerned with taking new ``chunks,'' and associating them with existing tracks, either
			// via overlap (box or box+mask), via ML, via bag of visual words, etc.
			// Q: how to map between a track and an Optional<Reconstruction>?
			// A: TODO
			//
			// Moreover, we should have some sort of quality threshold for beginning a reconstruction.
			// e.g., 6+ tiny frames may be needed for a distant car, but 2-3 ones could already be enough
			// for a much closer one...

			main_view->rgb->UpdateDeviceFromHost();
			main_view->depth->UpdateDeviceFromHost();

			// ``Graphically'' display the object tracks for debugging.
			for(const Track& track : this->instance_tracker_->GetTracks()) {
				cout << "Track: " << track.GetAsciiArt() << endl;
			}

			frame_idx_++;
		}

    ITMUChar4Image *InstanceReconstructor::GetInstancePreviewRGB(size_t track_idx) {
      const auto &tracks = instance_tracker_->GetTracks();
      if (tracks.empty()) {
        return nullptr;
      }

      size_t idx = track_idx;
      if (idx >= tracks.size()) {
        idx = tracks.size() - 1;
      }

      return tracks[idx].GetLastFrame().instance_view.GetView().rgb;
    }
	}
}

