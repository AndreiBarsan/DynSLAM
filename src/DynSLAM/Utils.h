#ifndef DYNSLAM_UTILS_H
#define DYNSLAM_UTILS_H

#include <string>
#include <sys/stat.h>

#include <highgui.h>

#include "DepthEngine.h"

namespace dynslam {
namespace utils {

bool EndsWith(const std::string &value, const std::string &ending);

/// \brief Streamlined 'sprintf' functionality for C++.
std::string Format(const std::string& fmt, ...);

inline bool file_exists(const std::__cxx11::string& name) {
  struct stat buffer;
  return stat(name.c_str(), &buffer) == 0;
}

} // namespace utils
} // namespace dynslam

#endif //DYNSLAM_UTILS_H
