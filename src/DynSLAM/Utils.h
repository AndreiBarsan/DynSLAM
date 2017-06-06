#ifndef DYNSLAM_UTILS_H
#define DYNSLAM_UTILS_H

#include <string>
#include <sys/stat.h>

#include <highgui.h>
#include <sys/time.h>

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

/// \brief Returns the number of milliseconds since the start of the epoch.
int64_t GetTimeMs();

/// \brief Converts an OpenCV image type to a human-readable string.
std::string Type2Str(int type);

// TODO(andrei): Consider moving the timing code to its own file.

/// \brief A simple multi-lap timer. All timestamps are expressed in milliseconds.
class Timer {
 public:
  Timer(const std::string &name) : name_(name), start_(-1), end_(-1), is_running_(false) { }

  void Reset() {
    start_ = -1;
    end_ = -1;
    laps_.clear();
    is_running_ = false;
  }

  void Start() {
    Reset();
    is_running_ = true;
    start_ = GetTimeMs();
  }

  void Lap() {
    assert(IsRunning());

    laps_.push_back(GetTimeMs());
  }

  void Stop() {
    assert(IsRunning());

    Lap();
    is_running_ = false;
    end_ = GetTimeMs();
  }

  bool IsRunning() const {
    return is_running_;
  }

  int64_t GetElapsed() const {
    assert(is_running_ && "Cannot get elapsed time of non-running timer. If the timer was already "
        "stopped, then please use 'GetDuration()'.");

    int64_t now = GetTimeMs();
    return now - start_;
  }

  int64_t GetDuration() const {
    assert(!is_running_ && "Cannot get duration of running timer. For running timers, please use "
        "'GetElapsed()'.");

    return end_ - start_;
  }

  double GetMeanLapTime() const {
    assert(laps_.size() > 0 && "Cannot compute the mean lap time if there are no laps.");

    double sum = 0.0;
    for(int64_t lap_time : laps_) {
      sum += (static_cast<double>(lap_time) - start_);
    }
    return sum / laps_.size();
  }

  const std::vector<int64_t> GetLaps() const {
    return laps_;
  }

  const std::string GetName() const {
    return name_;
  }

 private:
  std::string name_;
  int64_t start_;
  int64_t end_;
  bool is_running_;
  std::vector<int64_t> laps_;
};

/// \brief Helper for easily timing things. NOT thread-safe.
class Timers {
 private:
  static Timers instance_;

 public:
  static Timers& Get() {
    return instance_;
  }

  bool ContainsTimer(const std::string &name) {
    return timers_.find(name) != timers_.cend();
  }

  Timer& GetTimer(const std::string &name) {
    return timers_.at(name);
  }

  void Start(const std::string &name) {
    if (! ContainsTimer(name)) {
      timers_.emplace(std::make_pair(
          name, Timer(name)
      ));
    }

    timers_.at(name).Start();
    latest_name_ = name;
  }

  void Stop(const std::string &name) {
    timers_.at(name).Stop();
  }

  int64_t GetDuration(const std::string &name) {
    return timers_.at(name).GetDuration();
  }

  const std::string& GetLatestName() const {
    assert(timers_.size() > 0 && "No timers started.");
    return latest_name_;
  }

 private:
  Timers() { }
  std::map<std::string, Timer> timers_;
  std::string latest_name_;
};

/// \brief Easily start a timer.
void Tic(const std::string &name);

/// \brief Easily stop a timer and get the total measured duration.
int64_t Toc(const std::string &name, bool quiet = false);

/// \brief Stops the most recent timer and gets the total measured duration.
int64_t Toc(bool quiet = false);


} // namespace utils
} // namespace dynslam

#endif //DYNSLAM_UTILS_H

