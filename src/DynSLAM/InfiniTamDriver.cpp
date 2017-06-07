

#include "InfiniTamDriver.h"

namespace dynslam {
namespace drivers {

ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix, bool flip_ud, bool fix_roll) {
  Matrix4f M;
  for(int i = 0; i < 16; ++i) {
    M.m[i] = static_cast<float>(pangolin_matrix.m[i]);
  }

  // This code is convoluted and doesn't actually seem to help improve the UX too much...
  /*
  double yaw = atan2(pangolin_matrix(2, 1), pangolin_matrix(2, 2));
  double pitch = atan2(-pangolin_matrix(2, 0),
                       sqrt(pangolin_matrix(2, 1) * pangolin_matrix(2, 1) +
                            pangolin_matrix(2, 2) * pangolin_matrix(2, 2)));
  double roll = atan2(pangolin_matrix(1, 0), pangolin_matrix(0, 0));

  if (flip_ud) {
    // Fixes the unintuitive behaviour caused by the differences between ITM's raycaster and the
    // default 3D inspector from Pangolin.
//    yaw *= -1.0;
  }

  if (fix_roll) {
    // Prevent the camera from tilting sideways, since it's not useful when inspecting 3D street
    // reconstructions.
//    roll = 0.0;
  }

  Matrix3f new_x;
  new_x(0, 0) = 1; new_x(0, 1) = 0; new_x(0, 2) = 0;
  new_x(1, 0) = 0; new_x(1, 1) = cos(yaw); new_x(1, 2) = -sin(yaw);
  new_x(2, 0) = 0; new_x(2, 1) = sin(yaw); new_x(2, 2) = cos(yaw);

  Matrix3f new_y;
  new_y(0, 0) = cos(pitch); new_y(0, 1) = 0; new_y(0, 2) = sin(pitch);
  new_y(1, 0) = 0; new_y(1, 1) = 1; new_y(1, 2) = 0;
  new_y(2, 0) = -sin(pitch); new_y(2, 1) = 0; new_y(2, 2) = cos(pitch);

  Matrix3f new_z;
  new_z(0, 0) = cos(roll); new_z(0, 1) = -sin(roll); new_z(0, 2) = 0;
  new_z(1, 0) = sin(roll); new_z(1, 1) = cos(roll); new_z(0, 2) = 0;
  new_z(2, 0) = 0; new_z(2, 1) = 0; new_z(2, 2) = 1;
   */

  ITMPose itm_pose;
  itm_pose.SetM(M);
//  itm_pose.SetR(new_z * new_y * new_x);
  itm_pose.Coerce();

  return itm_pose;
}

void InfiniTamDriver::GetImage(ITMUChar4Image *out,
                               ITMMainEngine::GetImageType get_image_type,
                               const pangolin::OpenGlMatrix &model_view) {
  if (nullptr != this->view) {
    ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    ITMIntrinsics intrinsics = this->view->calib->intrinsics_d;
    ITMMainEngine::GetImage(
        out,
        get_image_type,
        &itm_freeview_pose,
        &intrinsics);
  }
  // Otherwise: We're before the very first frame, so no raycast is available yet.
}

} // namespace drivers}
} // namespace dynslam
