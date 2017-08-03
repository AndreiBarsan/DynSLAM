
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

//  const uchar *rendered_depthmap = dyn_slam->GetStaticMapRaycastPreview(
//      pango_pose,
//      PreviewType::kDepth
//  );
  const float *rendered_depthmap = dyn_slam->GetStaticMapRaycastDepthPreview(pango_pose);
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
//  uchar input_depthmap_uc[width * height * 4];
  // TODO(andrei): Use the affine depth calib params here
//  float input_pixels_to_meters = 1 / 1000.0f;

//  for(int row = 0; row < height; ++row) {
//    for(int col = 0; col < width; ++col) {
//      // The (signed) short-valued depth map encodes the depth expressed in millimeters.
//      short in_depth = input_depthmap->at<short>(row, col);
//      uchar byte_depth;
//      if (in_depth == std::numeric_limits<short>::max()) {
//        byte_depth = 0;
//      }
//      else {
//        byte_depth = static_cast<uchar>(round(in_depth * input_pixels_to_meters / max_depth_meters * 255));
//      }
//
//      int idx = (row * width + col) * 4;
//      input_depthmap_uc[idx] =     byte_depth;
//      input_depthmap_uc[idx + 1] = byte_depth;
//      input_depthmap_uc[idx + 2] = byte_depth;
//      input_depthmap_uc[idx + 3] = byte_depth;
//    }
//  }

  // TODO(andrei): Once you get this working, wrap images in OpenCV objects for readability and flexibility.
  std::vector<DepthEvaluation> evals;
  for(uint delta_max = 0; delta_max <= 15; ++delta_max) {
    evals.push_back(EvaluateDepth(lidar_pointcloud,
                                  rendered_depthmap,
                                  *input_depthmap,
                                  velodyne_->velodyne_to_rgb,
                                  dyn_slam->GetLeftRgbProjectionMatrix().cast<double>(),
                                  dyn_slam->GetRightRgbProjectionMatrix().cast<double>(),
                                  width,
                                  height,
                                  min_depth_meters,
                                  max_depth_meters,
                                  delta_max,
                                  1,
                                  nullptr));
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());
  return DepthFrameEvaluation(std::move(meta), max_depth_meters, std::move(evals));
}

DepthEvaluation Evaluation::EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                          const float *const rendered_depth,
                                          const cv::Mat1s &input_depth,
                                          const Eigen::Matrix4d &velo_to_left_gray_cam,
                                          const Eigen::Matrix34d &proj_left_color,
                                          const Eigen::Matrix34d &proj_right_color,
                                          const int frame_width,
                                          const int frame_height,
                                          const float min_depth_meters,
                                          const float max_depth_meters,
                                          const uint delta_max,
                                          const uint rendered_stride,
                                          ILidarEvalCallback *callback) const {
  const int kTargetTypeRange = std::numeric_limits<uchar>::max();
  long missing_rendered = 0;
  long errors_rendered = 0;
  long correct_rendered = 0;
  long missing_input = 0;
  long errors_input = 0;
  long correct_input = 0;
  long measurements = 0;

  const float left_focal_length_px = static_cast<float>(proj_left_color(0, 0));
  // TODO parameter
  const float baseline_m = 0.537150654273f;

//  cout << "EvaluateDepth [" << min_depth_meters << "--" << max_depth_meters << "]" << endl;

  int epi_errors = 0;
  int too_big_disps = 0;

  for (int i = 0; i < lidar_points.rows(); ++i) {
    Eigen::Vector4d velo_point = lidar_points.row(i).cast<double>();
    // Ignore the reflectance; we only care about 3D homogeneous coordinates.
    velo_point(3) = 1.0f;
    Eigen::Vector4d cam_point = velo_to_left_gray_cam * velo_point;
    cam_point /= cam_point(3);

    double velo_z = cam_point(2);

    if (velo_z < min_depth_meters || velo_z > max_depth_meters) {
      continue;
    }

    double velo_z_scaled = (velo_z / max_depth_meters) * kTargetTypeRange;
    assert(velo_z_scaled > 0 && velo_z_scaled <= kTargetTypeRange);
    uchar velo_z_uc = static_cast<uchar>(velo_z_scaled);

    Eigen::Vector3d velo_2d_left = proj_left_color * cam_point;
    Eigen::Vector3d velo_2d_right = proj_right_color * cam_point;
    velo_2d_left /= velo_2d_left(2);
    velo_2d_right /= velo_2d_right(2);

    // TODO(andrei): Compute LIDAR disparity (!), then extract raycast and input disparities (!,
    // i.e., not depths) and compare.

    // We ignore LIDAR points which fall outside the camera's retina.
    if (velo_2d_left(0) < 0 || velo_2d_left(0) >= frame_width ||
        velo_2d_left(1) < 0 || velo_2d_left(1) >= frame_height) {
      continue;
    }

    int row_left = static_cast<int>(round(velo_2d_left(1)));
    int col_left = static_cast<int>(round(velo_2d_left(0)));
    int row_right = static_cast<int>(round(velo_2d_right(1)));
    int col_right = static_cast<int>(round(velo_2d_right(0)));

    if (row_left != row_right) {
      int delta = row_left - row_right;
      float fdelta = velo_2d_left(1) - velo_2d_right(1);

      if (abs(fdelta) > 1.0) {
        epi_errors++;
        cerr << "Warning: epipolar violation or something! Delta: " << delta << "; float: "
             << fdelta << endl;
        cerr << "Whereby left float row was: " << velo_2d_left(1) << " and right float row was: "
             << velo_2d_right(1) << endl;
      }
    }

    // "Manually" compute the disparity of a LIDAR reading.
    int lidar_disparity = col_left - col_right;

    int idx_in_rendered = (row_left * frame_width + col_left) * rendered_stride;
//    int idx_in_input = (row_left * frame_width + col_left) * input_stride;

    const float rendered_depth_val = rendered_depth[idx_in_rendered];
    const float input_depth_val = input_depth.at<short>(row_left, col_left) / 1000.0f;

    // Units of measurement: px = (m * px) / m;
    // TODO(andrei): Just pass disparity maps. (Will require some extra engineering.)
    const int rendered_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / rendered_depth_val));
    const int input_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / input_depth_val));

    // TODO(andrei): Flag for switching between depth comparisons and disparity ones.
//    uint rendered_delta = depth_delta(rendered_depth_val, velo_z_uc);
//    uint input_delta = depth_delta(input_depth_val, velo_z_uc);

    const uint rendered_disp_delta = static_cast<uint>(std::abs(rendered_disp - lidar_disparity));
    const uint input_disp_delta = static_cast<uint>(std::abs(input_disp - lidar_disparity));

    if (fabs(rendered_depth_val) < 1e-5) {
      missing_rendered++;
    }
    else if (rendered_disp_delta > delta_max) {
      errors_rendered++;
    }
    else {
      correct_rendered++;
    }

    if (input_depth_val == 0) {
      missing_input++;
    }
    else if (input_disp_delta > delta_max) {
      errors_input++;
    }
    else {
      correct_input++;
    }

    /*
    if (fabs(rendered_depth_val) < 1e-5) {
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
     */

    measurements++;

    if (nullptr != callback) {
      callback->ProcessItem(i, velo_2d_left, rendered_depth_val, input_depth_val, velo_z_uc, frame_width, frame_height);
    }
  }
//  cout << "Epipolar violations: " << epi_errors << " out of " << measurements << "." << endl;
  cout << "Found " << too_big_disps << " suspiciously large disparities." << endl;

  DepthResult rendered_result(measurements, errors_rendered, missing_rendered, correct_rendered);
  DepthResult input_result(measurements, errors_input, missing_input, correct_input);

  return DepthEvaluation(delta_max,
                         std::move(rendered_result),
                         std::move(input_result));
}

}
}
