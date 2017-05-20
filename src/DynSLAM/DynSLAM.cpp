
#include <iostream>

#include <opencv/cv.h>
#include <highgui.h>

namespace dynslam {
  using namespace std;

  void hello() {
    cout << "Hello, DynSLAM!" << endl;
    try {
      cv::Mat img = cv::imread("/home/andrei/Pictures/george.jpg");
      cout << "Dimensions: " << img.cols << "x" << img.rows << endl;
      cv::imshow("The Summer of George", img);
      cv::waitKey(0);

    } catch (const cv::Exception &exception) {
      cerr << "This was supposed to be the summer of George..." << endl;
      cerr << exception.msg << endl;
    }
  }
}

// TODO(andrei): Use gflags for clean, easy-to-extend flag support.
int main() {
  dynslam::hello();
}
