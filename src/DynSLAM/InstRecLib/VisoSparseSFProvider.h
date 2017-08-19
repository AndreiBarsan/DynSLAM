

#ifndef INSTRECLIB_VISOSSFPROVIDER_H
#define INSTRECLIB_VISOSSFPROVIDER_H

#include "../Utils.h"
#include "SparseSFProvider.h"

#include <opencv/highgui.h>

#include "../../libviso2/src/viso_stereo.h"

namespace instreclib {

Eigen::Matrix4d VisoToEigen(const Matrix &viso_matrix);

// Note that at this point, we also use libviso2, which is primarily a visual odometry library, as a
// library for scene flow computation, in order to understand the motion of the other vehicles in
// view.
class VisoSparseSFProvider : public SparseSFProvider {
 public:
  VisoSparseSFProvider(VisualOdometryStereo::parameters &stereo_vo_params)
    : stereo_vo_(new VisualOdometryStereo(stereo_vo_params)),
      matches_available_(false),
      latest_flow_(SparseSceneFlow())
  { }

  virtual ~VisoSparseSFProvider() {
    delete stereo_vo_;
  }

  // We ignore the previous view, since libviso2 stores the previous frame internally.
  void ComputeSparseSF(const ViewPair&,
                       const ViewPair &current_view) override;

  virtual bool FlowAvailable() const {
    return matches_available_;
  }

  virtual SparseSceneFlow& GetFlow() {
    assert(matches_available_ && "Last frame's matches are not available.");
    return latest_flow_;
  }

  Eigen::Matrix4f GetLatestMotion() const override {
    return VisoToEigen(stereo_vo_->getMotion()).cast<float>();

  }

  std::vector<double> ExtractMotion(const std::vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow,
                                    const std::vector<double> &initial_estimate) const override;

 private:
  VisualOdometryStereo *stereo_vo_;
  bool matches_available_;
  SparseSceneFlow latest_flow_;
};

}  // namespace instreclib

#endif  // INSTRECLIB_VISOSSFPROVIDER_H
