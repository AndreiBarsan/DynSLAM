

#include "PrecomputedSegmentationProvider.h"
#include "InstanceSegmentationResult.h"

// TODO(andrei): Get rid of this dependency.
#include "../../InfiniTAM/InfiniTAM/Utils/FileUtils.h"
#include "../Utils.h"
#include "../Input.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <opencv/highgui.h>
#include <GL/gl.h>

namespace instreclib {
namespace segmentation {

using namespace std;
using namespace instreclib::utils;
using namespace dynslam::utils;

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
    const std::string &base_img_fpath) {
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

    // Process the mask file. The mask area covers the edges of the bounding box, too.
    dynslam::utils::Tic("Read mask and convert");
    dynslam::utils::Tic("Read mask");
    uint8_t *mask_pixels = ReadMask(mask_in, bounding_box.GetWidth(), bounding_box.GetHeight());
    dynslam::utils::Toc();
    cv::Mat *mask_cv_mat = new cv::Mat(
        bounding_box.GetHeight(),
        bounding_box.GetWidth(),
        CV_8UC1,
        (void*) mask_pixels
    );
    auto mask = make_shared<Mask>(bounding_box, mask_cv_mat);
    dynslam::utils::Toc();

    // TODO(andrei): Consider maintaining some overlap--we could use the 1.2 mask for sending info
    // to the reconstruction and e.g., 1.0 for sending it to the static map. However, the ambiguous
    // band could maybe be flagged with a lower update weight.
    float mask_rescale_factor = 1.1f;
    mask->Rescale(mask_rescale_factor);
    detections.emplace_back(class_probability, class_id, mask, this->dataset_used);

    instance_idx++;
  }

  return detections;
}

shared_ptr<InstanceSegmentationResult> PrecomputedSegmentationProvider::SegmentFrame(const cv::Mat4b &rgb) {
  stringstream img_fpath_ss;
  img_fpath_ss << this->seg_folder_ << "/"
               << "cls_" << setfill('0') << setw(6) << this->frame_idx_ << ".png";
  const string img_fpath = img_fpath_ss.str();

  if (last_seg_preview_ == nullptr) {
    last_seg_preview_ = new cv::Mat3b(rgb.rows, rgb.cols);
  }

  // TODO(andrei): Do we need the unchanged part?
  *last_seg_preview_ = cv::imread(img_fpath, CV_LOAD_IMAGE_UNCHANGED);

  if (! last_seg_preview_->data) {
    throw std::runtime_error(Format(
        "Could not read segmentation preview image from file [%s].",
        img_fpath));
  }

  stringstream meta_img_ss;
  meta_img_ss << this->seg_folder_ << "/" << setfill('0') << setw(6) << this->frame_idx_ << ".png";
  vector<InstanceDetection> instance_detections = ReadInstanceInfo(meta_img_ss.str());

  // We read pre-computed segmentations off the disk, so we assume this is 0.
  long inference_time_ns = 0L;

  this->frame_idx_++;

  return make_shared<InstanceSegmentationResult>(
      dataset_used,
      instance_detections,
      inference_time_ns);
}

const cv::Mat3b* PrecomputedSegmentationProvider::GetSegResult() const {
  return this->last_seg_preview_;
}

cv::Mat3b* PrecomputedSegmentationProvider::GetSegResult() {
  return this->last_seg_preview_;
}

}  // namespace segmentation
}  // namespace instreclib
