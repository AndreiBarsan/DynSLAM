
#include "CsvWriter.h"

#include <iostream>

namespace dynslam {
namespace eval {

using namespace std;

void CsvWriter::Write(const ICsvSerializable &data) {
  if (! wrote_header_) {
    *output_ << data.GetHeader() << endl;
    wrote_header_ = true;
  }

  *output_ << data.GetData() << endl;
}

}
}
