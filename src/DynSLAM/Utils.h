#ifndef DYNSLAM_UTILS_H
#define DYNSLAM_UTILS_H

#include <cmath>
#include <map>
#include <string>
#include <sys/stat.h>
#include <stack>
#include <vector>

#include <Eigen/Core>

namespace dynslam {
namespace utils {

/// \brief Very, VERY simple optional object wrapper.
template<typename T>
class Option {
 public:
  Option() : value_(nullptr) { }
  Option(T *value) : value_(value) { }

  bool IsPresent() const {
    return value_ != nullptr;
  }

  T& operator*() {
    return Get();
  }

  const T& operator*() const {
    return Get();
  }

  T& Get() {
    assert(IsPresent() && "Cannot dereference an empty optional!");
    return *value_;
  }

  const T& Get() const {
    assert(IsPresent() && "Cannot dereference an empty optional!");
    return *value_;
  }

  virtual ~Option() {
    delete value_;
  }

  static Option<T> Empty() {
    return Option<T>();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  T *value_;
};

template<typename T>
std::ostream& operator<<(std::ostream &out, const Option<T> &option) {
  out << "Option(";
  if (option.IsPresent()) {
    out << *option;
  }
  else {
    out << "empty";
  }
  out << ")";
  return out;
}


bool EndsWith(const std::string &value, const std::string &ending);

/// \brief Streamlined 'sprintf' functionality for C++.
/// \note If you're getting gibberish printing strings, make sure you pass 'c_str()' to the
/// function, not the actual string object.
std::string Format(const std::string& fmt, ...);

inline bool FileExists(const std::string &fpath) {
  struct stat buffer;
  return stat(fpath.c_str(), &buffer) == 0;
}

/// \brief Returns the number of milliseconds since the start of the epoch.
int64_t GetTimeMs();

/// \brief Returns the number of microseconds since the start of the epoch.
int64_t GetTimeMicro();

/// \brief Converts an OpenCV image type to a human-readable string.
std::string Type2Str(int type);

// TODO(andrei): Consider moving the timing code to its own file.

/// \brief A simple multi-lap timer. All timestamps are expressed in microseconds unless otherwise
///        stated.
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
    start_ = GetTimeMicro();
  }

  void Lap() {
    if (! IsRunning()) {
      throw std::runtime_error(Format("Timer [%s] not running; cannot stop!", name_.c_str()));
    }

    laps_.push_back(GetTimeMicro());
  }

  void Stop() {
    if (! IsRunning()) {
      throw std::runtime_error(Format("Timer [%s] not running; cannot stop!", name_.c_str()));
    }

    Lap();
    is_running_ = false;
    end_ = GetTimeMicro();
  }

  bool IsRunning() const {
    return is_running_;
  }

  int64_t GetElapsed() const {
    assert(is_running_ && "Cannot get elapsed time of non-running timer. If the timer was already "
        "stopped, then please use 'GetDuration()'.");

    int64_t now = GetTimeMicro();
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

/// \brief Returns a filename-friendly date string, such as '2017-01-01'.
std::string GetDate();

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
    names_.push(name);
  }

  void Stop(const std::string &name) {
    timers_.at(name).Stop();
    // TODO(andrei): Is this way of functioning too confusing?
    if (name == names_.top()) {
      names_.pop();
    }
  }

  int64_t GetDuration(const std::string &name) {
    return timers_.at(name).GetDuration();
  }

  std::string GetLatestName() const {
    assert(timers_.size() > 0 && "No timers started.");
    return names_.top();
  }

 private:
  Timers() { }
  std::map<std::string, Timer> timers_;
  std::stack<std::string> names_;
};

/// \brief Helper for starting a timer.
void Tic(const std::string &name);

/// \brief Stops the specified timer and gets the total measured duration in milliseconds.
int64_t Toc(const std::string &name, bool quiet = false);

/// \brief Stops the specified timer and gets the total measured duration in microseconds.
int64_t TocMicro(const std::string &name, bool quiet = false);

/// \brief Stops the most recent timer and gets the total measured duration in milliseconds.
int64_t Toc(bool quiet = false);

/// \brief Stops the most recent timer and gets the total measured duration in microseconds.
int64_t TocMicro(bool quiet = false);

/// \brief Computes a relative pose rotation error using the metric from the KITTI odometry evaluation.
inline float RotationError(const Eigen::Matrix4f &pose_error) {
  float a = pose_error(0, 0);
  float b = pose_error(1, 1);
  float c = pose_error(2, 2);
  double d = 0.5 * (a + b + c - 1.0);
  return static_cast<float>(acos(std::max(std::min(d, 1.0), -1.0)));
}

/// \brief Computes a relative pose translation error using the metric from the KITTI odometry evaluation.
inline float TranslationError(const Eigen::Matrix4f &pose_error) {
  float dx = pose_error(0, 3);
  float dy = pose_error(1, 3);
  float dz = pose_error(2, 3);
  return sqrt(dx * dx + dy * dy + dz * dz);
}

/// \brief Converts the pixel coordinates into [-1, +1]-style OpenGL coordinates.
/// \note The GL coordinates range from (-1.0, -1.0) in the bottom-left, to (+1.0, +1.0) in the
///       top-right.
Eigen::Vector2f PixelsToGl(const Eigen::Vector2f &px, const Eigen::Vector2f &px_range,
                           const Eigen::Vector2f &view_bounds);


} // namespace utils
} // namespace dynslam

#endif //DYNSLAM_UTILS_H

