
#ifndef DYNSLAM_DSHANDLER3D_H
#define DYNSLAM_DSHANDLER3D_H

#include <pangolin/pangolin.h>

namespace dynslam {
namespace gui {

/// \brief Customized 3D navigation handler for interactive 3D map visualizations.
/// Doesn't try to do the fancy object-aware rotations that Pangolin's builtin handler attempts to
/// do, which is preferred when visualizing large reconstructions and mixed raycast-raster renders.
class DSHandler3D : public pangolin::Handler {
 public:
  DSHandler3D(pangolin::OpenGlRenderState *cam_state,
              pangolin::AxisDirection enforce_up,
              float trans_scale,
              float zoom_fraction)
    : pangolin::Handler(),
      eye(-0.3, 0.4, 3.0),
      direction(0, 0, -1),
      enforce_up_(enforce_up),
      cam_state_(cam_state),
      last_pos_{0.0f, 0.0f}
  {
    UpdateModelViewMatrix();
  }

  void MouseMotion(pangolin::View &view, int x, int y, int button_state) override;

  void Mouse(pangolin::View &view,
             pangolin::MouseButton button,
             int x,
             int y,
             bool pressed,
             int button_state) override;

 protected:
  void UpdateModelViewMatrix();

 private:
  Eigen::Vector3d eye;
  Eigen::Vector3d direction;

  pangolin::AxisDirection enforce_up_;
  pangolin::OpenGlRenderState *cam_state_;
  float last_pos_[2];

  float yaw_accum = 0.0f;
  float pitch_accum = 0.0f;
};

} // namespace gui
} // namespace dynslam

#endif //DYNSLAM_DSHANDLER3D_H
