#ifndef DYNSLAM_CSVWRITER_H
#define DYNSLAM_CSVWRITER_H

#include <string>

namespace dynslam {
namespace eval {

/// \brief Interface for poor man's serialization.
/// In the long run, it would be nice to use protobufs or something for this...
class ICsvSerializable {
 public:
  virtual ~ICsvSerializable() = default;

  // TODO-LOW(andrei): The correct C++ way of doing this is by just making this writable to an ostream.
  /// \brief Should return the field names in the same order as GetData, without a newline.
  virtual std::string GetHeader() const = 0;
  virtual std::string GetData() const = 0;
};

class CsvWriter {
 public:
  explicit CsvWriter(const std::string &output_fpath)
    : wrote_header_(false),
      output_(new std::ofstream(output_fpath))
  {}

  CsvWriter(const CsvWriter &) = delete;
  CsvWriter(CsvWriter &&) = delete;
  CsvWriter& operator=(const CsvWriter &) = delete;
  CsvWriter& operator=(CsvWriter &&) = delete;

  void Write(const ICsvSerializable &data) {
    if (! wrote_header_) {
      *output_ << data.GetHeader() << endl;
      wrote_header_ = true;
    }

    *output_ << data.GetData() << endl;
  }

  virtual ~CsvWriter() {
    delete output_;
  }

 private:
  bool wrote_header_;
  ostream *output_;
};

}
}


#endif //DYNSLAM_CSVWRITER_H
