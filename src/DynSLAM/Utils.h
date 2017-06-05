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

inline bool FileExists(const std::string &fpath) {
  struct stat buffer;
  return stat(fpath.c_str(), &buffer) == 0;
}

/// \brief Converts an OpenCV image type to a human-readable string.
std::string Type2Str(int type);

} // namespace utils
} // namespace dynslam

#endif //DYNSLAM_UTILS_H

