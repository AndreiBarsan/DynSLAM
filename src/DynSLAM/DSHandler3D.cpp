
#include "DSHandler3D.h"

#include <Eigen/Geometry>

namespace dynslam {
namespace gui {

using namespace pangolin;
using namespace std;

void DSHandler3D::UpdateModelViewMatrix() {
  Eigen::Vector3d target = eye + direction;
  cam_state_->SetModelViewMatrix(pangolin::ModelViewLookAt(eye[0], eye[1], eye[2],
                                 target[0], target[1], target[2],
                                 enforce_up_));
}

void DSHandler3D::MouseMotion(pangolin::View &view, int x, int y, int button_state) {

  const float delta[2] = {(float) x - last_pos_[0], (float) y - last_pos_[1]};
  const float mag = delta[0] * delta[0] + delta[1] * delta[1];

  if (mag < 50.0f * 50.0f) {
    const GLprecision *up = AxisDirectionVector[enforce_up_];
    if (button_state == MouseButtonLeft) {
      // Left Drag: in plane translate
      Eigen::Vector3d up_v(up[0], up[1], up[2]);

      float dx = delta[0];
      float dy = -delta[1];

      Eigen::Vector3d newMotionX = direction.cross(up_v).normalized() * dx * 0.05f * trans_rot_scale_;
      Eigen::Vector3d newMotionY = direction.cross(up_v).cross(direction).normalized() * dy * 0.05f * trans_rot_scale_;
      eye += newMotionX;
      eye += newMotionY;

      UpdateModelViewMatrix();

    }
    else if (button_state == MouseButtonRight) {
      GLprecision aboutx = -0.3 * trans_rot_scale_ * delta[0];
      GLprecision abouty = -0.3 * trans_rot_scale_ * delta[1];

      yaw_accum_ -= aboutx;
      pitch_accum_ -= abouty;

      while (yaw_accum_ < -360.0) { yaw_accum_ += 360.0f; }
      while (yaw_accum_ > +360.0) { yaw_accum_ -= 360.0f; }

      float pitch_lim = 88.9f;
      if (pitch_accum_ > pitch_lim) {
        pitch_accum_ = pitch_lim;
      }
      if (pitch_accum_ < -pitch_lim) {
        pitch_accum_ = -pitch_lim;
      }

      // Do NOT do a barrel roll. In fact, don't roll at all because it's not really useful when
      // flying through a scene.
      double roll = 0.0f;
      double TO_RAD = M_PI / 180.0f;
      // The ordering of the rotation angles is specific to InfiniTAM's axis conventions.
      Eigen::Quaterniond rot_quat = Eigen::AngleAxisd(roll * TO_RAD, Eigen::Vector3d::UnitX())
          * Eigen::AngleAxisd(yaw_accum_ * TO_RAD, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(pitch_accum_ * TO_RAD, Eigen::Vector3d::UnitZ());

      direction = rot_quat * Eigen::Vector3d(0.0f, 0.0f, 1.0f);

      UpdateModelViewMatrix();
    }
  }

  last_pos_[0] = (float) x;
  last_pos_[1] = (float) y;
}

void DSHandler3D::Mouse(pangolin::View &view,
                        pangolin::MouseButton button,
                        int x,
                        int y,
                        bool pressed,
                        int button_state) {
  last_pos_[0] = static_cast<float>(x);
  last_pos_[1] = static_cast<float>(y);

  if (pressed) {
    if (button == pangolin::MouseWheelUp) {
      eye += direction.normalized() * zoom_scale_;
    }
    else if(button == pangolin::MouseWheelDown) {
      eye -= direction.normalized() * zoom_scale_;
    }

    UpdateModelViewMatrix();
  }
}

} // namespace gui
} // namespace dynslam

