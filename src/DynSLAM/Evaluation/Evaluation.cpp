
#include "Evaluation.h"
#include "ILidarEvalCallback.h"

namespace dynslam {
namespace eval {

void Evaluation::EvaluateFrame(Input *input, DynSlam *dyn_slam) {

  // TODO(andrei): Raycast of static-only raycast vs. no-dynamic-mapping vs. lidar vs. depth map.
  // TODO(andrei): Raycast of static+dynamic raycast vs. no-dynamic-mapping vs. lidar vs. depth map
  // No-dynamic-mapping == our system but with no semantics, object tracking, etc., so just ITM on
  // stereo.

  if (dyn_slam->GetCurrentFrameNo() > 0) {
    cout << "Starting evaluation of current frame..." << endl;
    // TODO wrap this vector into its own class (also CSV-serializable)
    DepthFrameEvaluation evals = EvaluateFrame(input->GetCurrentFrame() - 1, input, dyn_slam);

    if (! wrote_header_) {
      *csv_dump_ << evals.GetHeader() << endl;
      wrote_header_ = true;
    }

    // Append each measurement from this frame into the CSV file
    *csv_dump_ << evals.GetData() << endl;

    // ...and print some quick info in real time as well.
    cout << "Evaluation complete:" << endl;
    bool missing_depths_are_errors = false;
    if(missing_depths_are_errors) {
      cout << "(Missing = errors)" << endl;
    }
    else {
      cout << "(Not counting missing as errors.)" << endl;
    }
    for (auto &eval : evals.evaluations) {
      cout << "Evaluation on frame #" << evals.meta.frame_idx << ", delta max = "
           << setw(2) << eval.delta_max
           << "   Fusion accuracy = " << setw(7) << setprecision(3)
           << eval.fused_result.GetCorrectPixelRatio(missing_depths_are_errors)
           << " @ " << eval.fused_result.correct_count << " correct px "
           << " | Input accuracy  = " <<  setw(7) << setprecision(3)
           << eval.input_result.GetCorrectPixelRatio(missing_depths_are_errors)
          << " @ " << eval.input_result.correct_count << " correct px "
           << endl;
    }
  }
}

DepthFrameEvaluation Evaluation::EvaluateFrame(int frame_idx,
                                               Input *input,
                                               DynSlam *dyn_slam
) {
  auto lidar_pointcloud = velodyne_->ReadFrame(frame_idx);

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
  // TODO(andrei): Use the affine depth calib params here
  float input_pixels_to_meters = 1 / 1000.0f;

  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      // The (signed) short-valued depth map encodes the depth expressed in millimeters.
      short in_depth = input_depthmap->at<short>(row, col);
      uchar byte_depth;
      if (in_depth == std::numeric_limits<short>::max()) {
        byte_depth = 0;
      }
      else {
        byte_depth = static_cast<uchar>(round(in_depth * input_pixels_to_meters / max_depth_meters * 255));
      }

      int idx = (row * width + col) * 4;
      input_depthmap_uc[idx] =     byte_depth;
      input_depthmap_uc[idx + 1] = byte_depth;
      input_depthmap_uc[idx + 2] = byte_depth;
      input_depthmap_uc[idx + 3] = byte_depth;
    }
  }

  std::vector<DepthEvaluation> evals;
  for(uint delta_max = 0; delta_max <= 10; ++delta_max) {
    evals.push_back(EvaluateDepth(lidar_pointcloud,
                                  rendered_depthmap,
                                  input_depthmap_uc,
                                  velodyne_->velodyne_to_rgb,
                                  dyn_slam->GetProjectionMatrix().cast<double>(),
                                  width,
                                  height,
                                  min_depth_meters,
                                  max_depth_meters,
                                  delta_max,
                                  4,
                                  4,
                                  nullptr));
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());
  return DepthFrameEvaluation(std::move(meta), max_depth_meters, std::move(evals));
}

// TODO(andrei): Support rendering onto texture which then gets saved OR just render for a particular
// fixed delta_max on the fly from the GUI (simpler, but would require some serious refactoring of
// this method, which would be nice anyway).
DepthEvaluation Evaluation::EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                          const uchar *const rendered_depth,
                                          const uchar *const input_depth,
                                          const Eigen::Matrix4d &velo_to_cam,
                                          const Eigen::Matrix<double, 3, 4> &cam_proj,
                                          const int frame_width,
                                          const int frame_height,
                                          const float min_depth_meters,
                                          const float max_depth_meters,
                                          const uint delta_max,
                                          const uint rendered_stride,
                                          const uint input_stride,
                                          ILidarEvalCallback *callback) const {
  const int kTargetTypeRange = std::numeric_limits<uchar>::max();
  long missing_rendered = 0;
  long errors_rendered = 0;
  long correct_rendered = 0;
  long missing_input = 0;
  long errors_input = 0;
  long correct_input = 0;
  long measurements = 0;

//  cout << "EvaluateDepth [" << min_depth_meters << "--" << max_depth_meters << "]" << endl;

  for (int i = 0; i < lidar_points.rows(); ++i) {
    // TODO extract maybe loop body as separate method? => Simpler visualization without code dupe
    Eigen::Vector4d velo_point = lidar_points.row(i).cast<double>();
    // Ignore the reflectance; we only care about 3D homogeneous coordinates.
    velo_point(3) = 1.0f;
    Eigen::Vector4d cam_point = velo_to_cam * velo_point;
    cam_point /= cam_point(3);

    double velo_z = cam_point(2);

    if (velo_z < min_depth_meters || velo_z > max_depth_meters) {
      continue;
    }

    double velo_z_scaled = (velo_z / max_depth_meters) * kTargetTypeRange;
    assert(velo_z_scaled > 0 && velo_z_scaled <= kTargetTypeRange);
    uchar velo_z_uc = static_cast<uchar>(velo_z_scaled);

    Eigen::Vector3d velo_2d = cam_proj * cam_point;
    velo_2d /= velo_2d(2);

    // We ignore LIDAR points which fall outside the camera's retina.
    if (velo_2d(0) < 0 || velo_2d(0) >= frame_width ||
        velo_2d(1) < 0 || velo_2d(1) >= frame_height) {
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

    if (nullptr != callback) {
      callback->ProcessItem(i, velo_2d, rendered_depth_val, input_depth_val, velo_z_uc, frame_width, frame_height);
    }
  }

  DepthResult rendered_result(measurements, errors_rendered, missing_rendered, correct_rendered);
  DepthResult input_result(measurements, errors_input, missing_input, correct_input);

  return DepthEvaluation(delta_max,
                         std::move(rendered_result),
                         std::move(input_result));
}

}
}
