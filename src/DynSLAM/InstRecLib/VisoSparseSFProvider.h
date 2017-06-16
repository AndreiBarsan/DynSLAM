

#ifndef INSTRECLIB_VISOSSFPROVIDER_H
#define INSTRECLIB_VISOSSFPROVIDER_H

#include "../Utils.h"
#include "SparseSFProvider.h"

#include <opencv/highgui.h>

#include "../../libviso2/src/viso_stereo.h"

namespace instreclib {

// TODO(andrei): Consider using the joint VO+SF work Peidong sent you (open source from TUM).
// Note that at this point, we use libviso2, which is primarily a visual odometry library, as a
// library for scene flow computation, in order to understand the motion of the other vehicles in
// view.
class VisoSparseSFProvider : public SparseSFProvider {
 public:
  VisoSparseSFProvider(VisualOdometryStereo::parameters &stereo_vo_params)
    : stereo_vo_(new VisualOdometryStereo(stereo_vo_params)),
      matches_available_(false) { }

  virtual ~VisoSparseSFProvider() {
    delete stereo_vo_;
  }

  // TODO do we still need to pass both? It seems viso keeps track of them internally anyway.
  void ComputeSparseSF(const ViewPair &_,
                       const ViewPair &current_view) override {
    using namespace std;
    using namespace dynslam::utils;

    // TODO(andrei): Is this safe? What if OpenCV represents the images differently?
    uint8_t *left_bytes = current_view.first->data;
    uint8_t *right_bytes = current_view.second->data;

    // TODO(andrei): Consider caching this char version in an umbrella object, in case other parts
    // of the engine need it...
    /// This is the "paranoid" way of doing things, slow, but robust to any weird memory layout
    /// incompatibilities between libviso2 and OpenCV.
//    const auto *l_img = current_view.first;
//    const auto *r_img = current_view.second;
//    uint8_t *left_bytes = new uint8_t[l_img->rows * l_img->cols];
//    uint8_t *right_bytes = new uint8_t[r_img->rows * r_img->cols];
//    for(int i = 0; i < l_img->rows; ++i) {
//      for(int j = 0; j < l_img->cols; ++j) {
//        // We can do this because the two image must have the same dimensions anyway.
//        left_bytes[i * l_img->cols + j] = l_img->at<uint8_t>(i, j);
//        right_bytes[i * l_img->cols + j] = r_img->at<uint8_t>(i, j);
//      }
//    }

//    cv::Mat_<uint8_t> mat(l_img->rows, l_img->cols, right_bytes);
//    cv::imshow("Reconstructed preview...", mat);
//    cv::waitKey(0);

    int dims[] = {
        current_view.first->cols,
        current_view.first->rows,
        current_view.first->cols
    };
//    Tic("Viso2 frame processing");
    bool viso2_success = stereo_vo_->process(left_bytes, right_bytes, dims);
//    Toc();

    if (! viso2_success) {
      // TODO(andrei): In the long run, handle these failures more gracefully.
      cerr << "viso2 could not estimate egomotion and scene flow! (OK for first frame)" << endl;
      matches_available_ = false;
    }
    else {
      // TODO(andrei): Ensure this is not horribly slow (the match vector can be very big, and the
      // compiler isn't guaranteed to optimize away the two copies this call implies).
      // TODO(andrei): Don't read this right after the first frame. SF needs at least 2 frames before
      // it can compute the motion estimates.
      Tic("get matches");

      // Just marshal the data from the viso-specific format to DynSLAM format.
      std::vector<RawFlow> flow;
      for (const Matcher::p_match &match : stereo_vo_->getRawMatches()) {
        flow.emplace_back(match.u1c, match.v1c, match.i1c, match.u2c, match.v2c, match.i2c,
                          match.u1p, match.v1p, match.i1p, match.u2p, match.v2p, match.i2p);
      }
      SparseSceneFlow new_flow = {flow};
      latest_flow_ = new_flow;

      matches_available_ = true;
//      cout << "viso2 success! " << latest_flow_.matches.size() << " matches found." << endl;
//      cout << "               " << stereo_vo_->getNumberOfInliers() << " inliers" << endl;
      Toc();
    }
  }

  virtual bool FlowAvailable() const {
    return matches_available_;
  }

  virtual SparseSceneFlow& GetFlow() {
    assert(matches_available_ && "Last frame's matches are not available.");
    return latest_flow_;
  }

  std::vector<double> ExtractMotion(const std::vector<RawFlow> &flow) const override {
    // TODO-LOW(andrei): Extract helper method for this.
    std::vector<Matcher::p_match> flow_viso;
    for(const RawFlow &f : flow) {
      flow_viso.push_back(Matcher::p_match(f.prev_left(0), f.prev_left(1), f.prev_left_idx,
                                           f.prev_right(0), f.prev_right(1), f.prev_right_idx,
                                           f.curr_left(0), f.curr_left(1), f.curr_left_idx,
                                           f.curr_right(0), f.curr_right(1), f.curr_right_idx));
    }
    return stereo_vo_->estimateMotion(flow_viso);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  VisualOdometryStereo *stereo_vo_;
  bool matches_available_;
  SparseSceneFlow latest_flow_;
};

}  // namespace instreclib

#endif  // INSTRECLIB_VISOSSFPROVIDER_H
