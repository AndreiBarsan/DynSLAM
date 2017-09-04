

#include "PrecomputedSegmentationProvider.h"
#include "InstanceSegmentationResult.h"

#include "../Utils.h"
#include "../Input.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace instreclib {
namespace segmentation {

using namespace std;
using namespace instreclib::utils;
using namespace dynslam::utils;

/// Rescale factors used for RGB and depth masking of object instances.
float kCopyMaskRescaleFactor = 1.00f;
float kDeleteMaskRescaleFactor = 1.2f;

/// \brief Rescale factor used for scene flow masking of instances.
/// Smaller => fewer sparse scene flow vectors from, e.g., the background behind the object.
float kConservativeMaskRescaleFactor = 0.97f;

/// \brief Reads a numpy text dump of an object's 2D binary segmentation mask.
///
/// \param np_txt_in Input stream from the numpy text file containing the mask.
/// \param width The mask width, which is computed from its bounding box.
/// \param height The mask height, also computed from its bounding box.
/// \return A row-major array containing the mask info. The caller takes ownership.
///
/// \note The numpy file is already organized in 2D (as many lines as rows, etc.), so the given
/// width and height are simply used as an additional sanity check.
uint8_t *ReadMask(std::istream &np_txt_in, const int width, const int height) {
  // TODO(andrei): This code is very slow---it takes ~16ms just to read in a single mask. In practice
  // it's OK, since the end goal is to do live interop with the Caffe segmentation tool anyway.
  int lines_read = 0;
  string line_buf;
  uint8_t *mask_data = new uint8_t[height * width];

  while (getline(np_txt_in, line_buf)) {
    if (lines_read >= height) {
      stringstream error_ss;
      error_ss << "Image height mismatch. Went over the given limit of " << height << ".";
      throw runtime_error(error_ss.str());
    }

    istringstream line_ss(line_buf);
    int col = 0;
    while (!line_ss.eof()) {
      if (col >= width) {
        throw runtime_error(Format(
            "Image width mismatch. Went over specified width of %d when reading column %d. ",
            width,
            col));
      }

      // TODO(andrei): Remove this once you update your dumping code to directly dump bytes.
      double val;
      line_ss >> val;
      mask_data[lines_read * width + col] = static_cast<uint8_t>(val);
      col++;
    }

    lines_read++;
  }

  return mask_data;
}

vector<InstanceDetection> PrecomputedSegmentationProvider::ReadInstanceInfo(
    const std::string &base_img_fpath
) {
  // Loop through all possible instance detection dumps for this frame.
  //
  // They are saved by the pre-segmentation tool as:
  //     '${base_img_fpath}.${instance_idx}.{result,mask}.txt'.
  //
  // The result file is one line with the format "[x1 y1 x2 y2 junk], probability, class".
  // The first part represents the bounding box of the detection.
  //
  // The mask file is a numpy text file containing the saved (boolean) mask created by the neural
  // network. Its size is exactly the size of the bounding box.

  // We ignore detections smaller than this since they are not in any way useful in 3D object
  // reconstruction.
  // (Bigger than 40x40 can mess up e.g., the hill sequence @ 25m range by ignoring detections of
  //  things which can actually corrupt the map.)
  int min_area = static_cast<int>(round(45 * 45 * input_scale_));

  int instance_idx = 0;
  vector<InstanceDetection> detections;
  while (true) {
    stringstream result_fpath;
    result_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx
                 << ".result.txt";

    stringstream mask_fpath;
    mask_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx << ".mask.txt";

    ifstream result_in(result_fpath.str());
    ifstream mask_in(mask_fpath.str());
    if (!(result_in.is_open() && mask_in.is_open())) {
      // No more detections to process.
      break;
    }

    // Process the result metadata file
    string result;
    getline(result_in, result);

    BoundingBox bounding_box;
    float class_probability;
    int class_id;
    sscanf(result.c_str(), "[%d %d %d %d %*d], %f, %d", &bounding_box.r.x0, &bounding_box.r.y0,
           &bounding_box.r.x1, &bounding_box.r.y1, &class_probability, &class_id);

    if (bounding_box.GetArea() > min_area) {
      // Process the mask file. The mask area covers the edges of the bounding box, too.
  //    dynslam::utils::Tic("Read mask and convert");
  //    dynslam::utils::Tic("Read mask");
      uint8_t *mask_pixels = ReadMask(mask_in, bounding_box.GetWidth(), bounding_box.GetHeight());
  //    dynslam::utils::Toc();
      cv::Mat1b *mask_cv_mat = new cv::Mat1b(
          bounding_box.GetHeight(),
          bounding_box.GetWidth(),
          mask_pixels
      );

      bounding_box.r.x0 = static_cast<int>(round(bounding_box.r.x0 / input_scale_));
      bounding_box.r.y0 = static_cast<int>(round(bounding_box.r.y0 / input_scale_));
      bounding_box.r.x1 = static_cast<int>(round(bounding_box.r.x1 / input_scale_));
      bounding_box.r.y1 = static_cast<int>(round(bounding_box.r.y1 / input_scale_));

      auto copy_mask = make_shared<Mask>(bounding_box, mask_cv_mat);
      auto delete_mask = make_shared<Mask>(*copy_mask);
      auto conservative_mask = make_shared<Mask>(*copy_mask);
  //    dynslam::utils::Toc();

      copy_mask->Rescale(kCopyMaskRescaleFactor);
      float del_scale = kDeleteMaskRescaleFactor;
      // Adapt rescaling for distant objects. Constant chosen empirically.
      if (bounding_box.GetArea() < min_area * 1.375) {
        del_scale *= 1.2f;
      }
      delete_mask->Rescale(del_scale);
      conservative_mask->Rescale(kConservativeMaskRescaleFactor);

      detections.emplace_back(class_probability, class_id, copy_mask, delete_mask, conservative_mask,
                              this->dataset_used);
    }
    instance_idx++;
  }

  return detections;
}

shared_ptr<InstanceSegmentationResult> PrecomputedSegmentationProvider::SegmentFrame(const cv::Mat3b &rgb) {
  if (last_seg_preview_ == nullptr) {
    last_seg_preview_ = new cv::Mat3b(rgb.rows, rgb.cols);
  }
  stringstream img_fpath_ss;
  img_fpath_ss << this->seg_folder_ << "/"
               << "cls_" << setfill('0') << setw(6) << this->frame_idx_ << ".png";
  const string img_fpath = img_fpath_ss.str();
  if (! dynslam::utils::FileExists(img_fpath)) {
    throw runtime_error(dynslam::utils::Format("Unable to find segmentation preview at [%s].",
                                               img_fpath.c_str()));
  }
  cv::Mat seg_preview = cv::imread(img_fpath);
  cv::resize(seg_preview, *last_seg_preview_, cv::Size(), 1.0 / input_scale_, 1.0 / input_scale_, cv::INTER_LINEAR);

  if (! last_seg_preview_->data || last_seg_preview_->cols == 0 || last_seg_preview_->rows == 0) {
    throw runtime_error(Format(
        "Could not read segmentation preview image from file [%s].",
        img_fpath.c_str()));
  }

  auto result = ReadSegmentation(this->frame_idx_);
  this->frame_idx_++;
  return result;
}

const cv::Mat3b* PrecomputedSegmentationProvider::GetSegResult() const {
  return this->last_seg_preview_;
}

cv::Mat3b* PrecomputedSegmentationProvider::GetSegResult() {
  return this->last_seg_preview_;
}

std::shared_ptr<InstanceSegmentationResult> PrecomputedSegmentationProvider::ReadSegmentation(int frame_idx) {
  stringstream meta_img_ss;
  meta_img_ss << this->seg_folder_ << "/" << setfill('0') << setw(6) << frame_idx << ".png";
  vector<InstanceDetection> instance_detections = ReadInstanceInfo(meta_img_ss.str());

  // We read pre-computed segmentations off the disk, so we assume this is 0.
  long inference_time_ns = 0L;

  return make_shared<InstanceSegmentationResult>(
      dataset_used,
      instance_detections,
      inference_time_ns);
}

}  // namespace segmentation
}  // namespace instreclib
