
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
  static Eigen::Vector3d GetTranslation(const pangolin::OpenGlRenderState &state) {
    Eigen::Vector3d trans;
    trans(0) = state.GetModelViewMatrix()(0, 3);
    trans(1) = state.GetModelViewMatrix()(1, 3);
    trans(2) = state.GetModelViewMatrix()(2, 3);
    return trans;
  }

  static Eigen::Vector3d GetDirection(const pangolin::OpenGlRenderState &state) {
    Eigen::Matrix4d mv = state.GetModelViewMatrix();
    return mv.block(0, 0, 3, 3) * Eigen::Vector3d(0, 0, 1.0);
  }

  static Eigen::Vector3d GetEuler(const Eigen::Matrix3d &rot) {
    return rot.eulerAngles(0, 1, 2);
  }

 public:
  DSHandler3D(pangolin::OpenGlRenderState *cam_state,
              pangolin::AxisDirection enforce_up,
              float trans_scale,
              float zoom_scale)
    : eye(GetTranslation(*cam_state)),
      direction(GetDirection(*cam_state)),
      enforce_up_(enforce_up),
      cam_state_(cam_state),
      last_pos_{0.0f, 0.0f},
      trans_rot_scale_{trans_scale},
      zoom_scale_{zoom_scale}
  {
    Eigen::Matrix4d mv = cam_state->GetModelViewMatrix();
    Eigen::Vector3d euler = GetEuler(mv.block(0, 0, 3, 3));

    yaw_accum_ = euler(1) + M_PI_2;
    if (yaw_accum_ > 2 * M_PI) {
      yaw_accum_ -= M_PI;
    }

    pitch_accum_ = euler(0);
    if (pitch_accum_ > M_PI_2) {
      pitch_accum_ = M_PI_2;
    }
    else if (pitch_accum_ < -M_PI_2) {
      pitch_accum_ = -M_PI_2;
    }

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

  float yaw_accum_ = 0.0f;
  float pitch_accum_ = 0.0f;

  float trans_rot_scale_;
  float zoom_scale_;
};

} // namespace gui
} // namespace dynslam

#endif //DYNSLAM_DSHANDLER3D_H
