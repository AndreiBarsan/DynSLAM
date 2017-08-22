#ifndef DYNSLAM_SEGMENTEDVISUALIZATIONCALLBACK_H
#define DYNSLAM_SEGMENTEDVISUALIZATIONCALLBACK_H

#include "SegmentedCallback.h"
#include "ErrorVisualizationCallback.h"

namespace dynslam {
namespace eval {

class SegmentedVisualizationCallback : public SegmentedCallback {
 public:
  SegmentedVisualizationCallback(
      float delta_max,
      bool visualize_input,
      const Eigen::Vector2f &output_pane_bounds,
      unsigned char *colors,
      float *vertices,
      instreclib::segmentation::InstanceSegmentationResult *frame_segmentation,
      instreclib::reconstruction::InstanceReconstructor *reconstructor,
      LidarAssociation mode = LidarAssociation::kStaticMap)
      : SegmentedCallback(frame_segmentation, reconstructor),
        visualizer_(delta_max,
                    visualize_input,
                    output_pane_bounds,
                    colors,
                    vertices),
        mode_(mode) {}

  void ProcessLidarPoint(int idx,
                         const Eigen::Vector3d &velo_2d_homo,
                         float rendered_disp,
                         float rendered_depth,
                         float input_disp,
                         float input_depth,
                         float velodyne_disp,
                         int frame_width,
                         int frame_height
  ) override {
    LidarAssociation association = GetPointAssociation(velo_2d_homo);

    if (association == mode_) {
      visualizer_.ProcessLidarPoint(idx, velo_2d_homo, rendered_disp, rendered_depth, input_disp, input_depth, velodyne_disp, frame_width, frame_height);
    }
  }

  void Render() {
    visualizer_.Render();
  }

 private:
  ErrorVisualizationCallback visualizer_;
  const LidarAssociation mode_;
};

}
}

#endif //DYNSLAM_SEGMENTEDVISUALIZATIONCALLBACK_H

