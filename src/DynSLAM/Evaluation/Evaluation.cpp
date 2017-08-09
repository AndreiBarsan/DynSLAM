
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

  const float *rendered_depthmap = dyn_slam->GetStaticMapRaycastDepthPreview(pango_pose);
  auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
  auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
  input->GetFrameCvImages(frame_idx, input_rgb, input_depthmap);

  int width = dyn_slam->GetInputWidth();
  int height = dyn_slam->GetInputHeight();
  float min_depth_meters = input->GetDepthProvider()->GetMinDepthMeters();
  float max_depth_meters = input->GetDepthProvider()->GetMaxDepthMeters();

  // TODO(andrei): Once you get this working, wrap images in OpenCV objects for readability and flexibility.
  std::vector<DepthEvaluation> evals;
  const int kLargestMaxDelta = 12;
  bool compare_on_intersection = true;
  for(uint delta_max = 1; delta_max <= kLargestMaxDelta; ++delta_max) {
    evals.push_back(EvaluateDepth(lidar_pointcloud,
                                  rendered_depthmap,
                                  *input_depthmap,
                                  velodyne_->velodyne_to_rgb,
                                  dyn_slam->GetLeftRgbProjectionMatrix().cast<double>(),
                                  dyn_slam->GetRightRgbProjectionMatrix().cast<double>(),
                                  dyn_slam->GetStereoBaseline(),
                                  width,
                                  height,
                                  min_depth_meters,
                                  max_depth_meters,
                                  delta_max,
                                  compare_on_intersection,
                                  nullptr));
  }

  DepthEvaluationMeta meta(frame_idx, input->GetDatasetIdentifier());
  return DepthFrameEvaluation(std::move(meta), max_depth_meters, std::move(evals));
}

DepthEvaluation Evaluation::EvaluateDepth(const Eigen::MatrixX4f &lidar_points,
                                          const float *const rendered_depth,
                                          const cv::Mat1s &input_depth_mm,
                                          const Eigen::Matrix4d &velo_to_left_gray_cam,
                                          const Eigen::Matrix34d &proj_left_color,
                                          const Eigen::Matrix34d &proj_right_color,
                                          const float baseline_m,
                                          const int frame_width,
                                          const int frame_height,
                                          const float min_depth_meters,
                                          const float max_depth_meters,
                                          const uint delta_max,
                                          const bool compare_on_intersection,
                                          ILidarEvalCallback *callback) const {
  const float left_focal_length_px = static_cast<float>(proj_left_color(0, 0));
  long missing_rendered = 0;
  long errors_rendered = 0;
  long correct_rendered = 0;
  long missing_input = 0;
  long errors_input = 0;
  long correct_input = 0;
  long measurements = 0;

  int epi_errors = 0;

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

    Eigen::Vector3d velo_2d_left = proj_left_color * cam_point;
    Eigen::Vector3d velo_2d_right = proj_right_color * cam_point;
    velo_2d_left /= velo_2d_left(2);
    velo_2d_right /= velo_2d_right(2);

    int row_left = static_cast<int>(round(velo_2d_left(1)));
    int col_left = static_cast<int>(round(velo_2d_left(0)));
    int row_right = static_cast<int>(round(velo_2d_right(1)));
//    int col_right = static_cast<int>(round(velo_2d_right(0)));

    // We ignore LIDAR points which fall outside the left camera's retina.
    if (col_left < 0 || col_left >= frame_width ||
        row_left < 0 || row_left >= frame_height) {
      continue;
    }

    if (row_left != row_right) {
      int delta = row_left - row_right;
      float fdelta = velo_2d_left(1) - velo_2d_right(1);

      // Note that the LIDAR pointclouds aren't perfectly aligned, so this could occasionally happen.
      // In particular, this can happen when the car passes large trucks very closely, it seems.
      if (abs(fdelta) > 1.2) {
        epi_errors++;
        cerr << "Warning: epipolar violation! Delta: " << delta << "; float: "
             << fdelta << endl;
        cerr << "Whereby left float row was: " << velo_2d_left(1) << " and right float row was: "
             << velo_2d_right(1) << endl;
      }
    }

    const float lidar_disp = velo_2d_left(0) - velo_2d_right(0);
    if (lidar_disp < 0.0f) {
      throw std::runtime_error("Negative disparity in ground truth.");
    }

    int idx_in_rendered = row_left * frame_width + col_left;
    const float rendered_depth_m = rendered_depth[idx_in_rendered];
    const float input_depth_m = input_depth_mm.at<short>(row_left, col_left) / 1000.0f;
    if (input_depth_mm.at<short>(row_left, col_left) < 0) {
      cerr << "WARNING: Input depth negative value of " << input_depth_mm.at<short>(row_left, col_left)
           << " at "  << row_left << ", " << col_left << "." << endl;
    }

    // Old code which rounded before computing the disparity; slightly less accurate. We don't ever
    // have to round, actually!
//    const int rendered_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / rendered_depth_m));
//    const uint rendered_disp_delta = static_cast<uint>(std::abs(rendered_disp - lidar_disp));
//    const int input_disp = static_cast<int>(round((baseline_m * left_focal_length_px) / input_depth_val));
//    const uint input_disp_delta = static_cast<uint>(std::abs(input_disp - lidar_disp));

    // Units of measurement: px = (m * px) / m;
    const float rendered_disp = baseline_m * left_focal_length_px / rendered_depth_m;
    const float rendered_disp_delta = fabs(rendered_disp - lidar_disp);
    const float input_disp = baseline_m * left_focal_length_px / input_depth_m;
    const float input_disp_delta = fabs(input_disp - lidar_disp);

    /// We want to compare the fusion and the input map only where they're both present, since
    /// otherwise the fusion covers a much larger area, so its evaluation is tougher.
    // experimental metric
    if (compare_on_intersection && (fabs(input_depth_m) < 1e-5 || fabs(rendered_depth_m) < 1e-5)) {
      missing_input++;
      missing_rendered++;
    }
    else {
      if (fabs(input_depth_m) < 1e-5) {
        missing_input++;
      } else {
//        if (input_disp_delta > delta_max && (input_disp_delta > 0.05 * lidar_disp)) {
        if (input_disp_delta > delta_max) {
          errors_input++;
        } else {
          correct_input++;
        }
      }

      if (rendered_depth_m < 1e-5) {
        missing_rendered++;
      } else {
//        if (rendered_disp_delta > delta_max && (rendered_disp_delta > 0.05 * lidar_disp)) {
        if (rendered_disp_delta > delta_max) {
          errors_rendered++;
        } else {
          correct_rendered++;
        }
      }
    }


    /*
    if (fabs(rendered_depth_m) < 1e-5) {
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
      callback->LidarPoint(i,
                           velo_2d_left,
                           rendered_disp,
                           rendered_depth_m,
                           input_disp,
                           input_depth_m,
                           lidar_disp,
                           frame_width,
                           frame_height);
    }

  }

  if (epi_errors > 5) {
    cerr << "WARNING: Found " << epi_errors << " possible epipolar violations in the ground truth, "
         << "out of " << measurements << "." << endl;
  }

  DepthResult rendered_result(measurements, errors_rendered, missing_rendered, correct_rendered);
  DepthResult input_result(measurements, errors_input, missing_input, correct_input);

  return DepthEvaluation(delta_max,
                         std::move(rendered_result),
                         std::move(input_result));
}

}
}
