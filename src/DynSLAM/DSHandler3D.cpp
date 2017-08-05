
#include "DSHandler3D.h"

#include <Eigen/Geometry>

namespace dynslam {
namespace gui {

void DSHandler3D::UpdateModelViewMatrix() {
  Eigen::Vector3d target = eye + direction;
  cam_state->SetModelViewMatrix(pangolin::ModelViewLookAt(eye[0], eye[1], eye[2],
                                 target[0], target[1], target[2],
                                 enforce_up));
}

void DSHandler3D::MouseMotion(pangolin::View &view, int x, int y, int button_state) {
//    pangolin::Handler3D::MouseMotion(view, x, y, button_state);
  // Hack: copied Handler3D's method and modified it directly. There has to be a better way to
  // make this nicer...
  using namespace pangolin;

  const GLprecision rf = 0.01;
  const float delta[2] = {(float) x - last_pos[0], (float) y - last_pos[1]};
  const float mag = delta[0] * delta[0] + delta[1] * delta[1];

  if (mag < 50.0f * 50.0f) {
    OpenGlMatrix &mv = cam_state->GetModelViewMatrix();

    GLprecision current_trans[3];
    LieSetTranslation(mv.m, current_trans);

    const GLprecision *up = AxisDirectionVector[enforce_up];
//    GLprecision T_nc[3 * 4];
//    LieSetIdentity(T_nc);
//    bool rotation_changed = false;

    if (button_state == MouseButtonMiddle) {
      // ignore for now

    } else if (button_state == MouseButtonLeft) {
      // Left Drag: in plane translate

      // Act like WASD
      Eigen::Vector3d up_v(up[0], up[1], up[2]);

      float dx = delta[0];
      float dy = delta[1];

      Eigen::Vector3d newMotionX = direction.cross(up_v).normalized() * dx * 0.05f;
      Eigen::Vector3d newMotionY = direction.cross(up_v).cross(direction).normalized() * dy * 0.05f;
      eye += newMotionX;
      eye += newMotionY;


      UpdateModelViewMatrix();



      /*
      if (ValidWinDepth(last_z)) {
        GLprecision np[3];
        PixelUnproject(view, x, y, last_z, np);
        float kDragSpeedupFactor = 2.5f;
//          const GLprecision t[] = { np[0] - rot_center[0], np[1] - rot_center[1], 0};
        const GLprecision t[] = {(-np[0] + rot_center[0]) * 100 * kDragSpeedupFactor * tf,
                                 (np[1] - rot_center[1]) * 100 * kDragSpeedupFactor * tf,
                                 0};
        LieSetTranslation<>(T_nc, t);
        std::copy(np, np + 3, rot_center);
      } else {
//          const GLprecision t[] = { -10*delta[0]*tf, 10*delta[1]*tf, 0};
        const GLprecision t[] = {10 * delta[0] * tf, 10 * delta[1] * tf, 0};    // Removed a -
        LieSetTranslation<>(T_nc, t);
      }
       */
    } else if (button_state == MouseButtonRight) {
      /*
      GLprecision aboutx = -rf * delta[1];
      GLprecision abouty = -rf * delta[0];

      if (enforce_up) {
        // Special case if view direction is parallel to up vector
        const GLprecision updotz = mv.m[2] * up[0] + mv.m[6] * up[1] + mv.m[10] * up[2];
        if (updotz > 0.98) aboutx = std::min(aboutx, (GLprecision) 0.0);
        if (updotz < -0.98) aboutx = std::max(aboutx, (GLprecision) 0.0);
        // Module rotation around y so we don't spin too fast!
        abouty *= (1 - 0.6 * fabs(updotz));
      }

      // Right Drag: object centric rotation
      GLprecision T_2c[3 * 4];
      Rotation<>(T_2c, aboutx, abouty, (GLprecision) 0.0);
      GLprecision mrotc[3];
      MatMul<3, 1>(mrotc, rot_center, (GLprecision) -1.0);
      LieApplySO3<>(T_2c + (3 * 3), T_2c, mrotc);
      GLprecision T_n2[3 * 4];
      LieSetIdentity<>(T_n2);
      LieSetTranslation<>(T_n2, rot_center);
      LieMulSE3(T_nc, T_n2, T_2c);
      rotation_changed = true;
       */
    }

//    LieMul4x4bySE3<>(mv.m, T_nc, mv.m);

//    if (enforce_up != AxisNone && rotation_changed) {
//      EnforceUpT_cw(mv.m, up);
//    }
  }

  last_pos[0] = (float) x;
  last_pos[1] = (float) y;
}

void DSHandler3D::Mouse(pangolin::View &view,
                        pangolin::MouseButton button,
                        int x,
                        int y,
                        bool pressed,
                        int button_state) {
  // We flip the logic related to the right button being pressed: we want regular zoom when no
  // button is pressed, and direction-sensitive zoom if the RMB is pressed.
  if (button_state & pangolin::MouseButtonRight) {
    button_state &= ~(pangolin::MouseButtonRight);
  } else {
    button_state |= pangolin::MouseButtonRight;
  }

  last_pos[0] = static_cast<float>(x);
  last_pos[1] = static_cast<float>(y);

  if (pressed) {
    if (button == pangolin::MouseWheelUp) {
      eye += direction.normalized() * 0.5;
    }
    else if(button == pangolin::MouseWheelDown) {
      eye -= direction.normalized() * 0.5;
    }

    UpdateModelViewMatrix();
  }


  // would otherwise handle scrolling
//  pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);
}

} // namespace gui
} // namespace dynslam

