

#ifndef INSTRECLIB_VISOSSFPROVIDER_H
#define INSTRECLIB_VISOSSFPROVIDER_H

#include "SparseSFProvider.h"

#include <opencv/highgui.h>

#include "../../libviso2/src/viso_stereo.h"

namespace instreclib {

// Note that at this point, we use libviso2, which is primarily a visual odometry library, as a
// library for scene flow computation, in order to understand the motion of the other vehicles in
// view.
class VisoSparseSFProvider : public SparseSFProvider {
 public:
  VisoSparseSFProvider(VisualOdometryStereo::parameters &stereo_vo_params)
    : stereo_vo(new VisualOdometryStereo(stereo_vo_params)),
      matches_available_(false) { }

  virtual ~VisoSparseSFProvider() {
    delete stereo_vo;
  }

  // TODO do we still need to path both? It seems viso keeps track of them internally anyway.
  void ComputeSparseSceneFlow(const ViewPair &_,
                              const ViewPair &current_view) override {

    using namespace std;

    cout << "Computing sparse scene flow using libviso2 (TODO)..." << endl;

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
    bool viso2_success = stereo_vo->process(left_bytes, right_bytes, dims);
    if (! viso2_success) {
      // TODO(andrei): In the long run, handle these failures more gracefully.
//      throw runtime_error("viso2 could not estimate egomotion and scene flow!");
      cerr << "viso2 could not estimate egomotion and scene flow! (OK fir first frame)" << endl;
      matches_available_ = false;
    }
    else {
      // TODO(andrei): Ensure this is not horribly slow (the match vector can be very big, and the
      // compiler isn't guaranteed to optimize away the two copies this call implies).
      // TODO(andrei): Don't read this right after the first frame. SF needs at least 2 frames before
      // it can compute the motion estimates.
      matches_ = stereo_vo->getMatches();
      matches_available_ = true;
      cout << "viso2 success! " << matches_.size() << " matches found." << endl;
    }
  }

  bool MatchesAvailable() {
    return matches_available_;
  }

  const std::vector<Matcher::p_match>& GetMatches() const {
    assert(matches_available_ && "Last frame's matches are not available.");
    return matches_;
  }

 private:
  VisualOdometryStereo *stereo_vo;
  bool matches_available_;
  std::vector<Matcher::p_match> matches_;

};

}  // namespace instreclib

#endif  // INSTRECLIB_VISOSSFPROVIDER_H
