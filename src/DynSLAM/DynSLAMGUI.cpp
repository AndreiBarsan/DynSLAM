
#include <iostream>

#include <pangolin/pangolin.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <backward.hpp>

#include "../InfiniTAM/InfiniTAM/ORUtils/CUDADefines.h"

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
// TODO(andrei): this is a hack, please remove or depend on backward directly.
backward::SignalHandling sh;

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
    width = 640;
    height = 480;

    CreatePangolinDisplays();
    SetupDummyImage();
  }

  virtual ~PangolinGui() {
    delete dummy_image_texture;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // TODO move to own method
    pangolin::CreateWindowAndBind("DynSLAM GUI", UI_WIDTH + width * 2, 480);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1,
                                                             1000);

    float aspect_ratio = static_cast<float>(width) / height;
    pangolin::View &rgb_view = pangolin::Display("rgb").SetAspect(aspect_ratio);
    pangolin::View &depth_view = pangolin::Display("depth").SetAspect(aspect_ratio);
    pangolin::View &segment_view = pangolin::Display("segment").SetAspect(aspect_ratio);
    pangolin::View &object_view = pangolin::Display("object").SetAspect(aspect_ratio);

    pangolin::View &main_view = pangolin::Display("main");
    pangolin::View &detail_views = pangolin::Display("detail");

    main_view.SetAspect(aspect_ratio);
    main_view_free_cam = pangolin::OpenGlRenderState(proj,
                                                     pangolin::ModelViewLookAt(0, 0, -2, 0, 0, 0,
                                                                               pangolin::AxisY));
    detail_views.SetAspect(aspect_ratio);

    // TODO(andrei): Maybe wrap these guys?
    main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
                        pangolin::Attach::Pix(UI_WIDTH + width));
    main_view.SetHandler(new pangolin::Handler3D(main_view_free_cam));

    detail_views.SetBounds(0.0, 1.0, pangolin::Attach::Pix(width + UI_WIDTH), 1.0);
    detail_views.SetLayout(pangolin::LayoutEqual)
      .AddDisplay(rgb_view)
      .AddDisplay(depth_view)
      .AddDisplay(segment_view)
      .AddDisplay(object_view);

    cv::flip(dummy_img, dummy_img, kCvFlipVertical);
    cout << "Dimensions: " << dummy_img.cols << "x" << dummy_img.rows << endl;

    const int george_width = dummy_img.cols;
    const int george_height = dummy_img.rows;
    this->dummy_image_texture = new pangolin::GlTexture(
      george_width, george_height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glColor3f(1.0, 1.0, 1.0);

      // TODO(andrei): Only use Pangolin camera if not PITA. Otherwise, can just base everything off
      // InfiniTAM's raycasting.
      main_view.Activate(main_view_free_cam);
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      pangolin::glDrawColouredCube();

      rgb_view.Activate();
      glColor3f(1.0, 1.0, 1.0);

      // Mess with George's bytes a little bit
      //use fast 4-byte alignment (default anyway) if possible
      glPixelStorei(GL_UNPACK_ALIGNMENT, (dummy_img.step & 3) ? 1 : 4);

      //set length of one complete row in data (doesn't need to equal img.cols)
      glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img.step / dummy_img.elemSize());
      dummy_image_texture->Upload(dummy_img.data, GL_BGR, GL_UNSIGNED_BYTE);

      dummy_image_texture->RenderToViewport();

      depth_view.Activate();
      glColor3f(1.0, 0.0, 0.0);
      dummy_image_texture->RenderToViewport();

      segment_view.Activate();
      glColor3f(0.0, 1.0, 0.0);
      dummy_image_texture->RenderToViewport();

      object_view.Activate();
      glColor3f(0.0, 0.0, 1.0);
      dummy_image_texture->RenderToViewport();

//      plotter.Activate();
//      float val = static_cast<float>(sin(t));
//      t += tinc;
//      log.Log(val);

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

protected:

  void CreatePangolinDisplays() {
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
  }

  void SetupDummyImage() {

    cout << "Loading George..." << endl;
    dummy_img = cv::imread("/home/andrei/Pictures/george.jpg");
    cout << "Done." << endl;

//    cv::Mat dummy_img = cv::imread("/home/andrei/Pictures/george.jpg");

//    // Mess with George's bytes a little bit
//    //use fast 4-byte alignment (default anyway) if possible
//    glPixelStorei(GL_UNPACK_ALIGNMENT, (dummy_img.step & 3) ? 1 : 4);
//
//    //set length of one complete row in data (doesn't need to equal img.cols)
//    glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img.step/dummy_img.elemSize());
  }

private:
  /// Input frame dimensions. They dictate the overall window size.
  int width, height;

//  pangolin::View& main_view;
//  pangolin::View& detail_views;
//  pangolin::View& rgb_view;
//  pangolin::View& depth_view;
//  pangolin::View& segment_view;
//  pangolin::View& object_view;

  pangolin::GlTexture *dummy_image_texture;
  pangolin::OpenGlRenderState main_view_free_cam;

  cv::Mat dummy_img;
};

}
}

int main(int argc, char **argv) {
  dynslam::gui::PangolinGui gui;
  gui.Run();
}

