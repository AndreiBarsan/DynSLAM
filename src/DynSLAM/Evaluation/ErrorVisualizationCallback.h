
#include "ILidarEvalCallback.h"

namespace dynslam {
namespace eval {

class ErrorVisualizationCallback : public ILidarEvalCallback {
 public:
  ErrorVisualizationCallback(uint delta_max,
                             bool visualize_input,
                             const Eigen::Vector2f &output_pane_bounds,
                             unsigned char *colors,
                             float *vertices
  )
      : delta_max_(delta_max),
        visualize_input_(visualize_input),
        output_pane_bounds_(output_pane_bounds),
        colors_(colors),
        vertices_(vertices)
  {}

  void LidarPoint(int idx,
                  const Eigen::Vector3d &velo_2d_homo,
                  int rendered_disp,
                  float rendered_depth,
                  int input_disp,
                  float input_depth,
                  int velodyne_disp,
                  int width,
                  int height) override;

  void Render();

 private:
  int idx_v = 0;
  int idx_c = 0;

  uint delta_max_;

  /// \brief Whether to visualize delta vs. input, or vs the fused depth map.
  bool visualize_input_;

  Eigen::Vector2f output_pane_bounds_;

  unsigned char *colors_;
  float *vertices_;
};

}
}

