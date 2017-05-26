#ifndef DYNSLAM_UTILS_H
#define DYNSLAM_UTILS_H

#include <string>

namespace dynslam { namespace utils {

bool EndsWith(const std::string &value, const std::string &ending);

/// \brief Streamlined 'sprintf' functionality for C++.
std::string Format(const std::string& fmt, ...);

}}


#endif //DYNSLAM_UTILS_H
