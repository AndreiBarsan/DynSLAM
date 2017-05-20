
#include <iostream>

#include <opencv/cv.h>
#include <highgui.h>

// TODO(andrei): Configure InfiniTAM's includes so we can be less verbose here.
#include "../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"


namespace dynslam {
  using namespace std;

  void hello() {
    cout << "Hello, DynSLAM!" << endl;
    try {
      cv::Mat img = cv::imread("/home/andrei/Pictures/george.jpg");
      cout << "Dimensions: " << img.cols << "x" << img.rows << endl;

      Vector2i img_size(640, 480);
      cout << "ITM Vector2i: " << img_size.x << ", " << img_size.y << endl;

      cv::imshow("The Summer of George", img);
      cv::waitKey(0);

//      ITMLib::Engine::ITMMainEngine engine(settings, callib, img_size);

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
