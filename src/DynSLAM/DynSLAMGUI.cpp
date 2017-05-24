
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

const int UI_WIDTH = 180;

class PangolinGui {
public:
  PangolinGui() {
    cout << "Pangolin GUI initialized." << endl;

    // TODO(andrei): Put usefult things here.
    // Load configuration data
//    pangolin::ParseVarsFile("app.cfg");
    // TODO(andrei): Set from input dataset.
    int width = 640;
    int height = 480;

    cv::Mat img = cv::imread("/home/andrei/Pictures/george.jpg");
    cv::flip(img, img, kCvFlipVertical);
    cout << "Dimensions: " << img.cols << "x" << img.rows << endl;

    pangolin::CreateWindowAndBind("DynSLAM GUI", UI_WIDTH + width * 2, 480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // GUI stuff
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<function<void(void)>> a_button("ui.Some Button", []() {
      cout << "Clicked the button!" << endl;
    });
    pangolin::Var<function<void(void)>> quite_button("ui.Quit Button", []() {
      pangolin::QuitAll();
    });

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000);

    float aspect_ratio = static_cast<float>(width) / height;
    pangolin::View& rgb_view = pangolin::Display("rgb").SetAspect(aspect_ratio);
    pangolin::View& depth_view = pangolin::Display("depth").SetAspect(aspect_ratio);
    pangolin::View& segment_view = pangolin::Display("segment").SetAspect(aspect_ratio);
    pangolin::View& object_view = pangolin::Display("object").SetAspect(aspect_ratio);

    pangolin::View& main_view = pangolin::Display("main").SetAspect(aspect_ratio);
    pangolin::OpenGlRenderState main_view_free_cam(proj, pangolin::ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin::AxisY));
    pangolin::View& detail_views = pangolin::Display("detail").SetAspect(aspect_ratio);

    // TODO(andrei): Maybe wrap these guys?
    main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), pangolin::Attach::Pix(UI_WIDTH + width));
    main_view.SetHandler(new pangolin::Handler3D(main_view_free_cam));

    detail_views.SetBounds(0.0, 1.0, pangolin::Attach::Pix(width + UI_WIDTH), 1.0);
    detail_views.SetLayout(pangolin::LayoutEqual)
        .AddDisplay(rgb_view)
        .AddDisplay(depth_view)
        .AddDisplay(segment_view)
        .AddDisplay(object_view);

    // Custom: Add a plot to one of the views
    // Data logger object
//    pangolin::DataLog log;

    // Optionally add named labels
//    std::vector<std::string> labels;
//    labels.push_back(std::string("sin(t)"));
//    log.SetLabels(labels);
//
//    float t = 0.0f;
//    const float tinc = 0.1f;
//    // OpenGL 'view' of data. We might have many views of the same data.
//    pangolin::Plotter plotter(&log,0.0f,4.0f*(float)M_PI/tinc,-2.0f,2.0f,(float)M_PI/(4.0f*tinc),0.5f);
//    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
//    plotter.Track("$i");
//    plotter.SetBackgroundColour(pangolin::Colour::White());
//    plotter.SetTickColour(pangolin::Colour::Black());
//    plotter.SetAxisColour(pangolin::Colour::Black());

    const int george_width =  img.cols;
    const int george_height = img.rows;
    pangolin::GlTexture imageTexture(george_width, george_height, GL_RGB, false, 0, GL_RGB,GL_UNSIGNED_BYTE);

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() ) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glColor3f(1.0,1.0,1.0);

      // TODO(andrei): Only use Pangolin camera if not PITA. Otherwise, can just base everything off
      // InfiniTAM's raycasting.
      main_view.Activate(main_view_free_cam);
      glColor4f(1.0f,1.0f,1.0f,1.0f);
      pangolin::glDrawColouredCube();

      rgb_view.Activate();
      glColor3f(1.0,1.0,1.0);

      // Mess with George's bytes a little bit
      //use fast 4-byte alignment (default anyway) if possible
      glPixelStorei(GL_UNPACK_ALIGNMENT, (img.step & 3) ? 1 : 4);

      //set length of one complete row in data (doesn't need to equal img.cols)
      glPixelStorei(GL_UNPACK_ROW_LENGTH, img.step/img.elemSize());

      imageTexture.Upload(img.data, GL_BGR, GL_UNSIGNED_BYTE);
      imageTexture.RenderToViewport();

      depth_view.Activate();
      imageTexture.RenderToViewport();

      segment_view.Activate();
      imageTexture.RenderToViewport();

      object_view.Activate();
      imageTexture.RenderToViewport();

//      plotter.Activate();
//      float val = static_cast<float>(sin(t));
//      t += tinc;
//      log.Log(val);

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

