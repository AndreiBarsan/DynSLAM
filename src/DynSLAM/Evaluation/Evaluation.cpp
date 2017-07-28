
#include "Evaluation.h"

namespace dynslam {
namespace eval {

void Evaluation::EvaluateFrame(Input *input, DynSlam *dyn_slam) {

  // TODO(andrei): Raycast of static-only raycast vs. no-dynamic-mapping vs. lidar vs. depth map.
  // TODO(andrei): Raycast of static+dynamic raycast vs. no-dynamic-mapping vs. lidar vs. depth map
  // No-dynamic-mapping == our system but with no semantics, object tracking, etc., so just ITM on
  // stereo.

  // => TODO(andrei): Plot error as fn of delta, the allowed error. (see Sengupta/Torr paper Fig 10).

  if (input->GetCurrentFrame() > 0) {
    cout << "Starting evaluation of current frame..." << endl;
    EvaluateFrame(input->GetCurrentFrame() - 1, input, dyn_slam);
    cout << "Evaluation complete." << endl;
  }
}

std::vector<DepthEvaluation> Evaluation::EvaluateFrame(int frame_idx,
                                                       Input *input,
                                                       DynSlam *dyn_slam) {
  auto lidar_pointcloud = velodyne->ReadFrame(frame_idx);

  // TODO(andrei): Pose history in dynslam; will be necessary in delayed evaluation, as well
  // as if you wanna preview cute little frustums!
  if (frame_idx != input->GetCurrentFrame() - 1) {
    throw runtime_error("Cannot yet access old poses for evaluation.");
  }
  Eigen::Matrix4f epose = dyn_slam->GetPose().inverse();
  auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());

  const uchar *rendered_depthmap = dyn_slam->GetStaticMapRaycastPreview(
      pango_pose,
      PreviewType::kDepth
  );
  auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
  auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);

  input->GetFrameCvImages(frame_idx, input_rgb, input_depthmap);

//    const cv::Mat1s *input_depthmap = dyn_slam->GetDepthPreview();
//        cv::Mat1b input_depthmap_uc(height_, width_);

  int width = dyn_slam->GetInputWidth();
  int height = dyn_slam->GetInputHeight();
  float min_depth_meters = input->GetDepthProvider()->GetMinDepthMeters();
  float max_depth_meters = input->GetDepthProvider()->GetMaxDepthMeters();

  // TODO(andrei): Don't waste memory...
  uchar input_depthmap_uc[width * height * 4];

  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      // The (signed) short-valued depth map encodes the depth expressed in millimeters.
      short in_depth = input_depthmap->at<short>(row, col);
      // TODO(andrei): Use the affine depth calib params here if needed...
      uchar byte_depth;
      if (in_depth == std::numeric_limits<short>::max()) {
        byte_depth = 0;
      }
      else {
        byte_depth = static_cast<uchar>(in_depth / 1000.0 / max_depth_meters * 255);
      }

//            input_depthmap_uc.at<uchar>(row, col) = byte_depth;

      int idx = (row * width + col) * 4;
      input_depthmap_uc[idx] =     byte_depth;
      input_depthmap_uc[idx + 1] = byte_depth;
      input_depthmap_uc[idx + 2] = byte_depth;
      input_depthmap_uc[idx + 3] = byte_depth;
    }
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());

  // TODO(andrei): Nicer loop, use produced results, etc.
  for(int delta_max = 1; delta_max <= 5; ++delta_max) {
    EvaluateDepth(
        meta,
        lidar_pointcloud,
        rendered_depthmap,
        input_depthmap_uc,
        velodyne->velodyne_to_rgb,
        velodyne->rgb_project,
        width,
        height,
        min_depth_meters,
        max_depth_meters,
        delta_max
    );
  }

  // TODO populate
  return std::vector<DepthEvaluation>();
}

DepthEvaluation Evaluation::EvaluateDepth(const DepthEvaluationMeta &meta,
                                          const Eigen::MatrixX4f &lidar_points,
                                          const uchar *const rendered_depth,
                                          const uchar *const input_depth,
                                          const Eigen::Matrix4f &velo_to_cam,
                                          const Eigen::MatrixXf &cam_proj,
                                          const int frame_width,
                                          const int frame_height,
                                          const float min_depth_meters,
                                          const float max_depth_meters,
                                          const uint delta_max,
                                          const uint rendered_stride,
                                          const uint input_stride) const {
  const int kTargetTypeRange = std::numeric_limits<uchar>::max();
  long missing_rendered = 0;
  long errors_rendered = 0;
  long correct_rendered = 0;
  long missing_input = 0;
  long errors_input = 0;
  long correct_input = 0;
  long measurements = 0;

  for (int i = 0; i < lidar_points.rows(); ++i) {
    // TODO extract maybe loop body as separate method? => Simpler visualization without code dupe
    Eigen::Vector4f velo_point = lidar_points.row(i);
    // Ignore the reflectance; we only care about 3D homogeneous coordinates.
    velo_point(3) = 1.0f;
    Eigen::Vector4f cam_point = velo_to_cam * velo_point;
    cam_point /= cam_point(3);

    float velo_z = cam_point(2);

    if (velo_z < min_depth_meters || velo_z > max_depth_meters) {
      continue;
    }

    float velo_z_scaled = (velo_z / max_depth_meters) * kTargetTypeRange;
    assert(velo_z_scaled > 0 && velo_z_scaled <= kTargetTypeRange);
    uchar velo_z_uc = static_cast<uchar>(velo_z_scaled);

    Eigen::Vector3f velo_2d = cam_proj * cam_point;
    velo_2d /= velo_2d(2);

    // We ignore LIDAR points which fall outside the camera's retina.
    if (velo_2d(0) < 0 || velo_2d(0) >= frame_width ||
        velo_2d(1) < 0 || velo_2d(1) >= frame_height
        ) {
      continue;
    }

    // TODO-LOW(andrei): Using bilinear interpolation can slightly increase the accuracy of the
    // evaluation.
    int row = static_cast<int>(round(velo_2d(1)));
    int col = static_cast<int>(round(velo_2d(0)));
    int idx_in_rendered = (row * frame_width + col) * rendered_stride;
    int idx_in_input = (row * frame_width + col) * input_stride;

    uchar rendered_depth_val = rendered_depth[idx_in_rendered];
    uchar input_depth_val = input_depth[idx_in_input];

    uint rendered_delta = depth_delta(rendered_depth_val, velo_z_uc);
    uint input_delta = depth_delta(input_depth_val, velo_z_uc);

    if (rendered_depth_val == 0) {
      missing_rendered++;
    } else if (rendered_delta > delta_max) {
      errors_rendered++;
    } else {
      correct_rendered++;
    }

    if (input_depth_val == 0) {
      missing_input++;
    } else if (input_delta > delta_max) {
      errors_input++;
    } else {
      correct_input++;
    }

    measurements++;
  }

  DepthResult rendered_result(measurements, errors_rendered, missing_rendered, correct_rendered);
  DepthResult input_result(measurements, errors_input, missing_input, correct_input);

  return DepthEvaluation(meta, delta_max, max_depth_meters, rendered_result, input_result);
}

}
}
