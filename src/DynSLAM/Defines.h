
#ifndef DYNSLAM_DSDEFINES_H
#define DYNSLAM_DSDEFINES_H

#include <Eigen/Core>

// Necessary for having Eigen types as fields, but a little more readable for people less familiar
// with Eigen. More information can be found here: https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
#define SUPPORT_EIGEN_FIELDS EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

namespace Eigen {
  /// Simplify code dealing with projection matrices.
  using Matrix34d = Matrix<double, 3, 4>;
  using Matrix34f = Matrix<float, 3, 4>;
}

#endif //DYNSLAM_DSDEFINES_H
