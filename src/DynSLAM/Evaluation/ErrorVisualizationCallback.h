
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

  void LidarPoint(int idx,
                  const Eigen::Vector3d &velo_2d_homo,
                  int rendered_disp,
                  float rendered_depth,
                  int input_disp,
                  float input_depth,
                  int velodyne_disp,
                  int width,
                  int height) override {
    Eigen::Matrix<uchar, 3, 1> color;

    uint target_disp_delta = static_cast<uint>(
        (visualize_input_) ? abs(input_disp - velodyne_disp)
                           : abs(rendered_disp - velodyne_disp));
    int target_val = (visualize_input_) ? input_disp : rendered_disp;
    if (input_disp < 0 && fabs(input_depth) > 1e-5) {
      throw std::runtime_error(utils::Format(
          "Cannot have negative disparities, but found input_disp = %d!", input_disp));
    }
    if (rendered_disp < 0 && fabs(rendered_depth) > 1e-5) {
      throw std::runtime_error(utils::Format(
          "Cannot have negative disparities, but found rendered_disp = %d.", rendered_disp));
    }

    if (target_val > 0) {
      if (target_disp_delta > delta_max_) {
        color(0) = std::min(255, static_cast<int>(100 + (target_disp_delta - delta_max_ - 1) * 10));
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
//    cout << "Should render " << idx_v / 2 << " points..." << endl;
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

