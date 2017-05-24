
#include <iostream>

#include <pangolin/pangolin.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

/// \brief Define these because OpenCV doesn't. Used in the `cv::flip` OpenCV function.
enum {
  kCvFlipVertical = 0,
  kCvFlipHorizontal = 1,
  kCvFlipBoth = -1
};

namespace dynslam {
namespace gui {

using namespace std;

class PangolinGui {
public:
  PangolinGui() {
    cout << "Pangolin GUI initialized." << endl;

    cv::Mat img = cv::imread("/home/andrei/Pictures/george.jpg");
    cv::flip(img, img, kCvFlipVertical);
    cout << "Dimensions: " << img.cols << "x" << img.rows << endl;

    pangolin::CreateWindowAndBind("DynSLAM GUI", 640, 480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000);
    pangolin::OpenGlRenderState s_cam(proj, pangolin::ModelViewLookAt(1,0.5,-2,0,0,0, pangolin::AxisY) );
    pangolin::OpenGlRenderState s_cam2(proj, pangolin::ModelViewLookAt(0,0,-2,0,0,0, pangolin::AxisY) );
    pangolin::OpenGlRenderState s_cam3(proj, pangolin::ModelViewLookAt(0,0,-2, 0,0,0, pangolin::AxisY) );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam1 = pangolin::Display("cam1")
        .SetAspect(640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& d_cam2 = pangolin::Display("cam2")
        .SetAspect(640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam2));

    pangolin::View& d_cam3 = pangolin::Display("cam3")
        .SetAspect(640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam3));

    pangolin::View& d_cam4 = pangolin::Display("cam4")
        .SetAspect(640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(s_cam2));

    pangolin::View& d_img1 = pangolin::Display("img1")
      .SetAspect(640.0f/480.0f);

//  pangolin::View& d_img2 = pangolin::Display("img2")
//    .SetAspect(640.0f/480.0f);

    // Custom: Add a plot to one of the views
    // Data logger object
    pangolin::DataLog log;

    // Optionally add named labels
    std::vector<std::string> labels;
    labels.push_back(std::string("sin(t)"));
    log.SetLabels(labels);

    float t = 0.0f;
    const float tinc = 0.1f;
    // OpenGL 'view' of data. We might have many views of the same data.
    pangolin::Plotter plotter(&log,0.0f,4.0f*(float)M_PI/tinc,-2.0f,2.0f,(float)M_PI/(4.0f*tinc),0.5f);
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    plotter.Track("$i");
    plotter.SetBackgroundColour(pangolin::Colour::White());
    plotter.SetTickColour(pangolin::Colour::Black());
    plotter.SetAxisColour(pangolin::Colour::Black());

    // LayoutEqual is an EXPERIMENTAL feature - it requires that all sub-displays
    // share the same aspect ratio, placing them in a raster fasion in the
    // viewport so as to maximise display size.
    pangolin::Display("multi")
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_cam1)
        .AddDisplay(d_img1)
        .AddDisplay(d_cam2)
        .AddDisplay(plotter)
        .AddDisplay(d_cam3)
        .AddDisplay(d_cam4);

    const int width =  img.cols;
    const int height = img.rows;
    pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB,GL_UNSIGNED_BYTE);

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() ) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glColor3f(1.0,1.0,1.0);

      d_cam1.Activate(s_cam);
      pangolin::glDrawColouredCube();

      d_cam2.Activate(s_cam2);
      pangolin::glDrawColouredCube();

      d_cam3.Activate(s_cam3);
      pangolin::glDrawColouredCube();

      d_cam4.Activate(s_cam2);
      pangolin::glDrawColouredCube();

      d_img1.Activate();
      glColor4f(1.0f,1.0f,1.0f,1.0f);

      // Mess with George's bytes a little bit
      //use fast 4-byte alignment (default anyway) if possible
      glPixelStorei(GL_UNPACK_ALIGNMENT, (img.step & 3) ? 1 : 4);

      //set length of one complete row in data (doesn't need to equal img.cols)
      glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step/img.elemSize());

      imageTexture.Upload(img.data, GL_BGR, GL_UNSIGNED_BYTE);
      imageTexture.RenderToViewport();

      plotter.Activate();
      float val = static_cast<float>(sin(t));
      t += tinc;
      log.Log(val);

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }
};

}
}

int main(int argc, char **argv) {
  dynslam::gui::PangolinGui gui;
}

