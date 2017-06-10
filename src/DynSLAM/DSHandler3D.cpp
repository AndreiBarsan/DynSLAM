
#include "DSHandler3D.h"

namespace dynslam {
namespace gui {

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
    const GLprecision *up = AxisDirectionVector[enforce_up];
    GLprecision T_nc[3 * 4];
    LieSetIdentity(T_nc);
    bool rotation_changed = false;

    if (button_state == MouseButtonMiddle) {
      // Middle Drag: Rotate around view

      // Try to correct for different coordinate conventions.
      GLprecision aboutx = -rf * delta[1];
      GLprecision abouty = rf * delta[0];
      OpenGlMatrix &pm = cam_state->GetProjectionMatrix();
      abouty *= -pm.m[2 * 4 + 3];

      Rotation<>(T_nc, aboutx, abouty, (GLprecision) 0.0);
    } else if (button_state == MouseButtonLeft) {
      // Left Drag: in plane translate
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
    } else if (button_state == MouseButtonRight) {
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
    }

    LieMul4x4bySE3<>(mv.m, T_nc, mv.m);

    if (enforce_up != AxisNone && rotation_changed) {
      EnforceUpT_cw(mv.m, up);
    }
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

  pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);
}

} // namespace gui
} // namespace dynslam

