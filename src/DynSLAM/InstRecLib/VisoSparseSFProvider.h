

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
  VisoSparseSFProvider(VisualOdometryStereo::parameters &stereo_vo_params) {
    stereo_vo = new VisualOdometryStereo(stereo_vo_params);
  }

  virtual ~VisoSparseSFProvider() {
    delete stereo_vo;
  }

  // TODO do we still need to path both? It seems viso keeps track of them internally anyway.
  void ComputeSparseSceneFlow(const ViewPair &_,
                              const ViewPair &current_view) override {

    using namespace std;

    cout << "Computing sparse scene flow using libviso2 (TODO)..." << endl;

    // TODO(andrei): Consider caching this in an umbrella object, in case other parts
    // of the engine need it...

    // TODO(andrei): Is this safe? What if OpenCV represents the images differently?
    uint8_t *left_bytes = current_view.first->data;
    uint8_t *right_bytes = current_view.second->data;

    int dims[] = {
        current_view.first->cols,
        current_view.first->rows,
        current_view.first->rows
    };
    bool viso2_success = stereo_vo->process(left_bytes, right_bytes, dims);
    if (! viso2_success) {
      // TODO(andrei): In the long run, handle these failures more gracefully.
//      throw runtime_error("viso2 could not estimate egomotion and scene flow!");
      cerr << "viso2 could not estimate egomotion and scene flow!";
    }

    cout << "Computed scene flow OK." << endl;

    // TODO(andrei): Ensure this is not horribly slow (the match vector can be very big, and the
    // compiler isn't guaranteed to optimize away the two copies this call implies).
    // TODO(andrei): Don't read this right after the first frame. SF needs at least 2 frames before
    // it can compute the motion estimates.
    vector<Matcher::p_match> matches = stereo_vo->getMatches();

    // Return matches, and pass them to the instance reconstructor together with the 2D segmentation.
  }

 private:
  VisualOdometryStereo *stereo_vo;


};

}  // namespace instreclib

#endif  // INSTRECLIB_VISOSSFPROVIDER_H
