#ifndef DYNSLAM_CSVWRITER_H
#define DYNSLAM_CSVWRITER_H

#include <fstream>
#include <string>
#include "../Utils.h"

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
  const std::string output_fpath_;

  explicit CsvWriter(const std::string &output_fpath)
    : output_fpath_(output_fpath),
      wrote_header_(false),
      output_(new std::ofstream(output_fpath))
  {
    if(! utils::FileExists(output_fpath)) {
      throw std::runtime_error("Could not open CSV file. Does the folder it should be in exist?");
    }
  }

  CsvWriter(const CsvWriter &) = delete;
  CsvWriter(CsvWriter &&) = delete;
  CsvWriter& operator=(const CsvWriter &) = delete;
  CsvWriter& operator=(CsvWriter &&) = delete;

  void Write(const ICsvSerializable &data);

  virtual ~CsvWriter() {
    delete output_;
  }

 private:
  bool wrote_header_;
  std::ostream *output_;
};

}
}


#endif //DYNSLAM_CSVWRITER_H
