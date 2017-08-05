
#ifndef DYNSLAM_DSHANDLER3D_H
#define DYNSLAM_DSHANDLER3D_H

#include <pangolin/pangolin.h>

namespace dynslam {
namespace gui {

/// \brief Customized 3D navigation handler for interactive 3D map visualizations.
class DSHandler3D : public pangolin::Handler3D {
 public:
  explicit DSHandler3D(pangolin::OpenGlRenderState &cam_state)
      : Handler3D(cam_state),
        eye(-0.3, 0.4, 3.0),
        direction(0, 0, -1)
  {}

  DSHandler3D(pangolin::OpenGlRenderState &cam_state,
              pangolin::AxisDirection enforce_up,
              float trans_scale,
              float zoom_fraction)
      : Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction),
        eye(-0.3, 0.4, 3.0),
        direction(0, 0, -1)
  {}

  void Keyboard(pangolin::View &view, unsigned char key, int x, int y, bool pressed) override {
    pangolin::Handler3D::Keyboard(view, key, x, y, pressed);
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
};

} // namespace gui
} // namespace dynslam

#endif //DYNSLAM_DSHANDLER3D_H
