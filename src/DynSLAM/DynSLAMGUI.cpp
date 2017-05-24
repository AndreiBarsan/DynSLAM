
#include <iostream>

#include <pangolin/pangolin.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <backward.hpp>
#include <GL/glut.h>

#include "../InfiniTAM/InfiniTAM/ORUtils/CUDADefines.h"
#include "ImageSourceEngine.h"


// TODO(andrei): Use [RIP] tags to signal spots where you wasted more than 30 minutes debugging a
// small, silly issue.

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

const int UI_WIDTH = 300;

// TODO(andrei): Move away from here.
/// \brief Main DynSLAM class interfacing between different submodules.
class DynSlam {

public:
    void Initialize(ITMMainEngine *itm_static_scene_engine_, ImageSourceEngine *image_source) {

      this->image_source_ = image_source;

      window_size_.x = image_source->getDepthImageSize().x;
      window_size_.y = image_source->getDepthImageSize().y;

      this->itm_static_scene_engine_ = itm_static_scene_engine_;
      this->current_frame_no_ = 0;

//      int fake_argc = 0;
//      glutInit(&fake_argc, nullptr);
//      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//      glutInitWindowSize(window_size_.x, window_size_.y);
//      glutCreateWindow("DynSLAM Lightweight GUI");

      // TODO(andrei): Multiple panel support, if you don't go QT.
//      glGenTextures(1, panel_texture_ids_);

//      glutDisplayFunc(DynSlamGlutGui::GlutDisplay);
//      glutKeyboardUpFunc(DynSlamGlutGui::GlutKeyboard);
//      glutIdleFunc(DynSlamGlutGui::GlutIdle);

      bool allocate_gpu = true;
      Vector2i input_shape = image_source->getDepthImageSize();
      out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_rgb_image_= new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_raw_depth_image_ = new ITMShortImage(input_shape, true, allocate_gpu);

      // TODO(andrei): Own CUDA safety wrapper. With blackjack. And hookers.
      ITMSafeCall(cudaThreadSynchronize());

      cout << "DynSLAM initialization complete." << endl;
    }

  void ProcessFrame() {
    if (! image_source_->hasMoreImages()) {
      cout << "No more frames left in image source." << endl;
      return;
    }

    // Read the images from the first part of the pipeline
    image_source_->getImages(input_rgb_image_, input_raw_depth_image_);

    // Forward them to InfiniTAM for the background reconstruction.
    itm_static_scene_engine_->ProcessFrame(input_rgb_image_, input_raw_depth_image_);
    ITMSafeCall(cudaThreadSynchronize());

    current_frame_no_++;
  }

  int last_frame = -1;
  const unsigned char* GetImageData() {
    // TODO(andrei): Improve this crap.
    // TODO(andrei): Get rid of reliance on itam enums.
    if (last_frame != current_frame_no_ ) {
      last_frame = current_frame_no_;
      cout << "slam img size: " << itm_static_scene_engine_->GetImageSize() << endl;
      itm_static_scene_engine_->GetImage(
        out_image_,
        ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);
    }

    return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
  }

  int GetInputWidth() {
    return image_source_->getDepthImageSize().width;
  }

  int GetInputHeight() {
    return image_source_->getDepthImageSize().height;
  }

  int GetCurrentFrameNo() {
    return current_frame_no_;
  }

private:
  ITMLibSettings itm_lib_settings_;
  // TODO(andrei): Write custom image source.
  ImageSourceEngine *image_source_;

  // This is the main reconstruction component. Should split for dynamic+static.
  // In the future, we may need to write our own.
  // For now, this shall only handle reconstructing the static part of a scene.
  ITMMainEngine *itm_static_scene_engine_;

  ITMUChar4Image *out_image_;
  ITMUChar4Image *input_rgb_image_;
  ITMShortImage  *input_raw_depth_image_;

  int current_frame_no_;

  Vector2i window_size_;

//  uint panel_texture_ids_[1];
//  bool needs_refresh_;
//  Action next_action_;
};

/// TODO(andrei): Seriously consider using QT or wxWidgets. Pangolin is VERY limited in terms of the
/// widgets it supports. It doesn't even seem to support multiline text, or any reasonable way to
/// paint labels or lists or anything... Might be better not to worry too much about this, since
/// there isn't that much time...
class PangolinGui {
public:
  PangolinGui(DynSlam *dyn_slam) : dyn_slam_(dyn_slam) {
    cout << "Pangolin GUI initialized." << endl;

    // TODO(andrei): Put useful things in this config.
    // Load configuration data
//    pangolin::ParseVarsFile("app.cfg");
    // TODO(andrei): Proper scaling to save space and memory.
    width = dyn_slam->GetInputWidth(); // / 1.5;
    height = dyn_slam->GetInputHeight();// / 1.5;

    SetupDummyImage();
    CreatePangolinDisplays();
  }

  virtual ~PangolinGui() {
    delete dummy_image_texture;
  }

  /// \brief Executes the main Pangolin input and rendering loop.
  void Run() {
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);

      // TODO(andrei): The buffers you're now allocating for intermediate results are very wasteful.
      // It's almost 800Mb of VRAM just for panel stuff. Please fix this!
      // TODO(andrei): Only use Pangolin camera if not PITA. Otherwise, can just base everything off
      // InfiniTAM's raycasting.
      main_view->Activate();
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      const unsigned char *slam_frame_data = dyn_slam_->GetImageData();

      if(dyn_slam_->GetCurrentFrameNo() > 0) {
        // Hack for inspecting stuff
        // In CV terms, InfiniTAM produces CV_8UC4 output.

//        cv::Mat mat(height, width, CV_8UC4, (void *) slam_frame_data);
//
//        cv::imshow("CRAWLING IN MY SKIIIIN", mat);
//        cv::waitKey(0);
      }

      // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance!
      slam_preview->Upload(slam_frame_data, GL_RGBA, GL_UNSIGNED_BYTE);
      slam_preview->RenderToViewport(true);

      rgb_view.Activate();
      glColor3f(1.0, 1.0, 1.0);

      // TODO(andrei): Undo these state changes, since they mess up the rest of the pipeline.
//      UploadDummyTexture();
//      dummy_image_texture->RenderToViewport();
//
//      depth_view.Activate();
//      glColor3f(1.0, 0.0, 0.0);
//      dummy_image_texture->RenderToViewport();
//
//      segment_view.Activate();
//      glColor3f(0.0, 1.0, 0.0);
//      dummy_image_texture->RenderToViewport();
//
//      object_view.Activate();
//      glColor3f(0.0, 0.0, 1.0);
//      dummy_image_texture->RenderToViewport();
//
//      glColor3f(1.0, 1.0, 1.0);
//      pangolin::GlFont::I().Text("No data available").Draw(0,0,0);

//      plotter.Activate();
//      float val = static_cast<float>(sin(t));
//      t += tinc;
//      log.Log(val);

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

protected:

  /// \brief Creates the GUI layout and widgets.
  void CreatePangolinDisplays() {
    pangolin::CreateWindowAndBind("DynSLAM GUI", UI_WIDTH + width, height * 2);

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
    pangolin::Var<function<void(void)>> quit_button("ui.Quit Button", []() {
      pangolin::QuitAll();
    });
    pangolin::Var<string> reconstructions("ui.Reconstructions", "<none>");

    pangolin::Var<function<void(void)>> previous_object("ui.Previous Object", []() {
      cout << "Will select previous reconstructed object, once available..." << endl;
    });
    pangolin::Var<function<void(void)>> next_object("ui.Next Object", []() {
      cout << "Will select next reconstructed object, once available..." << endl;
    });
    pangolin::RegisterKeyPressCallback('n', [&]() {
      cout << "Next frame should happen now!" << endl;
      dyn_slam_->ProcessFrame();
    });

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(
      width, height, 420, 420, width / 2, height / 2, 0.1, 1000);

    float aspect_ratio = static_cast<float>(width) / height;
    rgb_view = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view = pangolin::Display("depth").SetAspect(aspect_ratio);
    segment_view = pangolin::Display("segment").SetAspect(aspect_ratio);
    object_view = pangolin::Display("object").SetAspect(aspect_ratio);

    // Storing pointers to these objects prevents a series of strange issues. The objects remain
    // under Pangolin's management, so they don't need to be deleted by the current class.
    main_view = &(pangolin::Display("main"));
    detail_views = &(pangolin::Display("detail"));

    main_view->SetAspect(aspect_ratio);
//    main_view_free_cam = pangolin::OpenGlRenderState(
//      proj, pangolin::ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin::AxisY));
    detail_views->SetAspect(aspect_ratio);

    // TODO(andrei): Maybe wrap these guys?
    main_view->SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
                         pangolin::Attach::Pix(UI_WIDTH + width));
//    main_view->SetHandler(new pangolin::Handler3D(main_view_free_cam));

    detail_views->SetBounds(0.0, 1.0, pangolin::Attach::Pix(width + UI_WIDTH), 1.0);
    detail_views->SetLayout(pangolin::LayoutEqual)
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

    this->slam_preview = new pangolin::GlTexture(
      width, height, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);


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
  }

private:
  DynSlam *dyn_slam_;

  /// Input frame dimensions. They dictate the overall window size.
  int width, height;

  pangolin::View *main_view;
  pangolin::View *detail_views;
  pangolin::View rgb_view;
  pangolin::View depth_view;
  pangolin::View segment_view;
  pangolin::View object_view;

  pangolin::GlTexture *dummy_image_texture;
  pangolin::GlTexture *slam_preview;
//  pangolin::OpenGlRenderState main_view_free_cam;

  cv::Mat dummy_img;


  void UploadDummyTexture() {
    // Mess with George's bytes a little bit for OpenGL <-> OpenCV compatibility.
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (dummy_img.step & 3) ? 1 : 4);
    //set length of one complete row in data (doesn't need to equal img.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, dummy_img.step/dummy_img.elemSize());
    dummy_image_texture->Upload(dummy_img.data, GL_BGR, GL_UNSIGNED_BYTE);
  }
};

}
}

int main(int argc, char **argv) {
  using namespace dynslam;
  gui::DynSlam *dyn_slam = new gui::DynSlam();
  const string dataset_root = "/home/andrei/work/libelas/cmake-build-debug/odo_seq_06/";
  const string calib_fpath = dataset_root + "calib.txt";
  const string rgb_image_format = dataset_root + "Frames/%04i.ppm";
  const string depth_image_format = dataset_root + "Frames/%04i.pgm";
  ImageSourceEngine *image_source = new ImageFileReader(
    calib_fpath.c_str(),
    rgb_image_format.c_str(),
    depth_image_format.c_str()
  );

  ITMLibSettings *settings = new ITMLibSettings();

  ITMMainEngine *static_scene_engine = new ITMMainEngine(
    settings,
    new ITMRGBDCalib(image_source->calib),
    image_source->getRGBImageSize(),
    image_source->getDepthImageSize()
  );

  dyn_slam->Initialize(static_scene_engine, image_source);

  gui::PangolinGui pango_gui(dyn_slam);
  pango_gui.Run();

  delete dyn_slam;
}

