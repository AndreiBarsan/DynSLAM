

#include "Utils.h"

#include <iostream>
#include <cstdarg>
#include <cstring>
#include <memory>
#include <sys/time.h>

#include <opencv/cv.h>
#include <mutex>

namespace dynslam {
namespace utils {

using namespace std;

bool EndsWith(const string &value, const string &ending){
  if (ending.size() > value.size()) {
    return false;
  }

  return equal(ending.rbegin(), ending.rend(), value.rbegin());
}

string Format(const string& fmt, ...) {
  // TODO-LOW(andrei): Use varadic templates to implement in a typesafe, string-friendly manner.
  // Keeps track of the resulting string size.
  size_t out_size = fmt.size() * 2;
  unique_ptr<char[]> formatted;
  va_list ap;
  while (true) {
    formatted.reset(new char[out_size]);
    strcpy(&formatted[0], fmt.c_str());
    va_start(ap, fmt);
    int final_n = vsnprintf(&formatted[0], out_size, fmt.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || static_cast<size_t>(final_n) >= out_size) {
      int size_update = final_n - static_cast<int>(out_size) + 1;
      out_size += abs(size_update);
    }
    else {
      break;
    }
  }

  return string(formatted.get());
}

std::string Type2Str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

std::string GetDate() {
  time_t rawtime;
  tm * timeinfo;
  char today_s[200];
  time (&rawtime);
  timeinfo = localtime (&rawtime);
  strftime(today_s, 200, "%Y-%m-%d",timeinfo);

  return std::string(today_s);
}

Timers Timers::instance_;

int64_t MicroToMilli(int64_t time_micro) {
  return static_cast<int64_t>(round(time_micro / 1000.0));
}

int64_t GetTimeMs() {
  timeval time;
  gettimeofday(&time, nullptr);
  int64_t time_us = time.tv_sec * 1000L + time.tv_usec / 1000;
  return time_us;
}

int64_t GetTimeMicro() {
  timeval time;
  gettimeofday(&time, nullptr);
  int64_t time_us = time.tv_sec * 1000L * 1000L + time.tv_usec;
  return time_us;
}

void Tic(const std::string &name) {
  Timers::Get().Start(name);
}

int64_t Toc(const std::string &name, bool quiet) {
  assert(Timers::Get().ContainsTimer(name) && "Toc: Unknown timer. Are you timing things on multiple threads? That's not supported.");
  Timers::Get().Stop(name);
  int64_t duration_ms = MicroToMilli(Timers::Get().GetDuration(name));
  if (! quiet) {
    cout << "Timer: " << name << " took " << duration_ms << "ms." << endl;
  }
  return duration_ms;
}

int64_t TocMicro(const std::string &name, bool quiet) {
  assert(Timers::Get().ContainsTimer(name) && "TocMicro: Unknown timer. Are you timing things on multiple threads? That's not supported.");
  Timers::Get().Stop(name);
  int64_t duration_micro = Timers::Get().GetDuration(name);
  if (! quiet) {
    cout << "Timer: " << name << " took " << duration_micro << "μs." << endl;
  }
  return duration_micro;
}

int64_t Toc(bool quiet) {
  const string& name = Timers::Get().GetLatestName();
  return Toc(name, quiet);
}

int64_t TocMicro(bool quiet) {
  const string& name = Timers::Get().GetLatestName();
  return TocMicro(name, quiet);
}

Eigen::Vector2f PixelsToGl(const Eigen::Vector2f &px, const Eigen::Vector2f &px_range,
                           const Eigen::Vector2f &view_bounds) {
  float view_w = view_bounds(0);
  float view_h = view_bounds(1);

  float px_x = px(0) * (view_w / px_range(0));
  float px_y = px(1) * (view_h / px_range(1));

  float gl_x = ((px_x - view_w) / view_w + 0.5f) * 2.0f;
  // We need the opposite sign here since pixel coordinates are assumed to have the origin in the
  // top-left, while the GL origin is in the bottom-left.
  float gl_y = ((view_h - px_y) / view_h - 0.5f) * 2.0f;

  return Eigen::Vector2f(gl_x, gl_y);
}

} // namespace utils
} // namespace dynslam

