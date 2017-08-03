
#include "ILidarEvalCallback.h"

#include <algorithm>

#include <Eigen/Eigen>
#include <opencv/cv.h>
#include <GL/gl.h>

#include "../Utils.h"

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

  // TODO(andrei): Fix this to work with disparities
  void ProcessItem(int idx,
                   const Eigen::Vector3d &velo_2d_homo,
                   unsigned char rendered_depth,
                   unsigned char input_depth,
                   unsigned char velodyne_depth,
                   int width,
                   int height
  ) override {
    Eigen::Matrix<uchar, 3, 1> color;

    uint target_delta = static_cast<uint>(
        (visualize_input_) ? abs(input_depth - velodyne_depth)
                           : abs(rendered_depth - velodyne_depth));
    uchar target_val = (visualize_input_) ? input_depth : rendered_depth;

    if (target_val != 0) {
      if (target_delta > delta_max_) {
        color(0) = std::min(255, static_cast<int>(180 + (target_delta - delta_max_ - 1) * 10));
        color(1) = 40;
        color(2) = 40;
      } else {
        color(0) = 10;
        color(1) = 255; //- target_delta * 10;
        color(2) = 255; //- target_delta * 10;
      }
    }
    else {
      color(0) = 0;
      color(1) = 0;
      color(2) = 0;
    }

    Eigen::Vector2f frame_size(width, height);
    Eigen::Vector2f gl_pos = utils::PixelsToGl(Eigen::Vector2f(velo_2d_homo(0), velo_2d_homo(1)),
                                               frame_size, output_pane_bounds_);
    GLfloat x = gl_pos(0);
    GLfloat y = gl_pos(1);

    vertices_[idx_v++] = x;
    vertices_[idx_v++] = y;
    colors_[idx_c++] = color(0);
    colors_[idx_c++] = color(1);
    colors_[idx_c++] = color(2);
  }

  void Render() {
    glDisable(GL_DEPTH_TEST);
    pangolin::glDrawColoredVertices<float>(idx_v / 2, vertices_, colors_, GL_POINTS, 2, 3);
    glEnable(GL_DEPTH_TEST);
  }

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

