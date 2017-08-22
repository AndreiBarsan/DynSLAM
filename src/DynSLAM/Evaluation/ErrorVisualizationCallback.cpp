#include "ErrorVisualizationCallback.h"

#include <pangolin/pangolin.h>
#include <opencv/cv.h>

#include "../Utils.h"

namespace Eigen {
// Same convention as OpenCV's Mat1b.
using Vector3b = Matrix<uchar, 3, 1>;
}

namespace dynslam {
namespace eval {

/// \brief Adds a colored circle to the given vertex and color (CPU) buffers.
void AddCircle(
    float *vertex_data,
    uchar *color_data,
    int &idx_v,
    int &idx_c,
    const Eigen::Vector2f &pos,
    const Eigen::Vector3b &color,
    float radius,
    float aspect_ratio,
    int n_vertices = 25
) {
  const double slice_size = 2 * M_PI / n_vertices;
  for(int i = 0; i < n_vertices;++i) {
    float x_off = static_cast<float>(radius * cos(-i * slice_size));
    float y_off = static_cast<float>(radius * sin(-i * slice_size)) * aspect_ratio;
    vertex_data[idx_v++] = pos(0) + x_off;
    vertex_data[idx_v++] = pos(1) + y_off;
    color_data[idx_c++] = color(0);
    color_data[idx_c++] = color(1);
    color_data[idx_c++] = color(2);
  }
}

void AddPoint(
    float *vertex_data,
    uchar *color_data,
    int &idx_v,
    int &idx_c,
    const Eigen::Vector2f &pos,
    const Eigen::Vector3b &color
) {
  vertex_data[idx_v++] = pos(0);
  vertex_data[idx_v++] = pos(1);
  color_data[idx_c++] = color(0);
  color_data[idx_c++] = color(1);
  color_data[idx_c++] = color(2);
}

void ErrorVisualizationCallback::ProcessLidarPoint(
    int idx,
    const Eigen::Vector3d &velo_2d_homo,
    float rendered_disp,
    float rendered_depth,
    float input_disp,
    float input_depth,
    float velodyne_disp,
    int frame_width,
    int frame_height
) {
  Eigen::Vector3b color;

  float target_disp_delta = (float)((visualize_input_) ? fabs(input_disp - velodyne_disp)
                                                       : fabs(rendered_disp - velodyne_disp));
  int target_val = (visualize_input_) ? input_disp : rendered_disp;
  if (input_disp < 0 && fabs(input_depth) > 1e-5) {
    throw std::runtime_error(utils::Format(
        "Cannot have negative input disparities, but found input_disp = %d!", input_disp));
  }
  if (rendered_disp < 0 && fabs(rendered_depth) > 1e-5) {
//    throw std::runtime_error(utils::Format(
//        "Cannot have negative disparities, but found rendered_disp = %d; its corresponding "
//        "rendered depth was %.4f.",
//        rendered_disp,
//        rendered_depth
//    ));
    std::cerr << utils::Format(
        "Warning: Cannot have negative disparities, but found rendered_disp = %d; its corresponding "
            "rendered depth was %.4f.",
        rendered_disp,
        rendered_depth
    ) << std::endl;
  }

  if (target_val > 0) {
    if (target_disp_delta > delta_max_) {
      color(0) = std::min(255, static_cast<int>(150 + (target_disp_delta - delta_max_ - 1) * 5));
      color(1) = 60;
      color(2) = 60;
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

  Eigen::Vector2f frame_size(frame_width, frame_height);
  Eigen::Vector2f gl_pos = utils::PixelsToGl(Eigen::Vector2f(velo_2d_homo(0), velo_2d_homo(1)),
                                             frame_size, output_pane_bounds_);

  float aspect = output_pane_bounds_(0) * 1.0f / output_pane_bounds_(1);
  float radius = 0.5f / output_pane_bounds_(1);
  AddCircle(vertices_, colors_, idx_v, idx_c, gl_pos, color, radius, aspect);
}

void ErrorVisualizationCallback::Render() {
  glDisable(GL_DEPTH_TEST);
  int n_vertices_per_circle = 25;
  int circle_count = idx_v / (2 * n_vertices_per_circle);

  GLint *starting_elements = new GLint[circle_count];
  GLint *counts = new GLint[circle_count];

  for(int i = 0; i < circle_count; ++i) {
    starting_elements[i] = i * n_vertices_per_circle;
    counts[i] = n_vertices_per_circle;
  }

  int n_color_elements = 3;
  int n_vertex_elements = 2;

  glColorPointer(n_color_elements, GL_UNSIGNED_BYTE, 0, colors_);
  glEnableClientState(GL_COLOR_ARRAY);

  glVertexPointer(n_vertex_elements, GL_FLOAT, 0, vertices_);
  glEnableClientState(GL_VERTEX_ARRAY);

  // Inspired by Pangolin: draw both the filling and the outside to make the result smooth.
  glMultiDrawArrays(GL_TRIANGLE_FAN, starting_elements, counts, circle_count);
  glMultiDrawArrays(GL_LINE_STRIP, starting_elements, counts, circle_count);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glEnable(GL_DEPTH_TEST);
  delete[] starting_elements;
  delete[] counts;
}

}
}
