
#ifndef DYNSLAM_EVALUATION_H
#define DYNSLAM_EVALUATION_H

#include "../DynSlam.h"
#include "../Input.h"
#include "Velodyne.h"

namespace dynslam {
  class DynSlam;
}

namespace dynslam {
namespace eval {

/// \brief Main class handling the quantitative evaluation of the DynSLAM system.
class Evaluation {

 public:
  Evaluation(const std::string &dataset_root, const Input::Config &input_config)
    : velodyne(new Velodyne(
        utils::Format("%s/%s", dataset_root.c_str(), input_config.velodyne_folder.c_str()),
        input_config.velodyne_fname_format)) { }

  /// \brief Supermethod in charge of all per-frame evaluation metrics.
  void EvaluateFrame(Input *input, DynSlam *dyn_slam);

  virtual ~Evaluation() {
    delete velodyne;
  }

 private:
  Velodyne *velodyne;

};

}
}

#endif //DYNSLAM_EVALUATION_H
