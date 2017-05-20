
#include <iostream>

#include <opencv/cv.h>
#include <highgui.h>

#include "ImageSourceEngine.h"

// TODO(andrei): Configure InfiniTAM's includes so we can be less verbose here.
#include "../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"

#include <GL/glut.h>


// TODO(andrei): TODO file for code-specific TODOs.
// TODO(andrei): clang-format.

namespace dynslam {
  using namespace std;
  using namespace ITMLib::Objects;

  /// \brief A simple reconstruction GUI based on the InfiniTAM UIEngine.
  class DynSlamGlutGui {
  public:

    // TODO(andrei): Write own image source, since your requirements are different!
    void Initialize(ITMMainEngine *itm_static_scene_engine_, ImageSourceEngine *image_source) {
      // TODO(andrei): Can't this be a ctor?

      this->image_source_ = image_source;

      window_size_.x = image_source->getDepthImageSize().x;
      window_size_.y = image_source->getDepthImageSize().y;


      this->itm_static_scene_engine_ = itm_static_scene_engine_;
      this->current_frame_no_ = 0;

      // TODO(andrei): Do we need specific stuff for glut?
      glutInit(0, nullptr);
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
      glutInitWindowSize(window_size_.x, window_size_.y);
      glutCreateWindow("DynSLAM Lightweight GUI");

      // TODO(andrei): Multiple panel support, if you don't go QT.
      glGenTextures(1, panel_texture_ids_);

      glutDisplayFunc([&]() { this->GlutDisplay(); });
      glutKeyboardFunc([&](unsigned char key, int x, int y) {
          // space -> next frame
          cout << "Key up: " << key << ", x = " << x << ", y =" << y << endl;
      });
      glutIdleFunc([&]() { this->GlutIdle(); });

      bool allocate_gpu = true;
      Vector2i input_shape = image_source->getDepthImageSize();
      out_image_ = new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_rgb_image_= new ITMUChar4Image(input_shape, true, allocate_gpu);
      input_raw_depth_image_ = new ITMShortImage(input_shape, true, allocate_gpu);


      // TODO(andrei): Own safety wrapper. With blackjac. And hookers.
      ITMSafeCall(cudaThreadSynchronize());

      cout << "DynSLAM initialization complete." << endl;
    }

    void ProcessFrame() {
      if (! image_source_->hasMoreImages()) {
        cout << "No more frames left in image source." << endl;
        return;
      }

      itm_static_scene_engine_->ProcessFrame(input_rgb_image_, input_raw_depth_image_);
      ITMSafeCall(cudaThreadSynchronize());

      current_frame_no_++;
    }

    void Run() {
      glutMainLoop();
    }

    void Shutdown() {
      // TODO(andrei): Clean up after yourself.
    }

  private:
    ITMLibSettings itm_lib_settings_;
    // TODO(andrei): Write custom.
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

    uint panel_texture_ids_[1];

    void GlutDisplay() {
      itm_static_scene_engine_->GetImage(out_image_, ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);

      glClear(GL_COLOR_BUFFER_BIT);
      glColor3f(1.0f, 1.0f, 1.0f);

      // TODO finish writing

    }

    void GlutIdle() {
      cout << "GLUT Idle Yay!!" << endl;
    }

  };

  /// \brief Crude main loop running InfiniTAM.
  /// In the future, this will be refactored into multiple components.
  void ScaffoldingLoop() {
    auto gui = DynSlamGlutGui();
    // TODO pass as arg

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

    gui.Initialize(static_scene_engine, image_source);
  }

  void George() {
    cout << "Hello, DynSLAM!" << endl;
    try {
      cv::Mat img = cv::imread("/home/andrei/Pictures/george.jpg");
      cout << "Dimensions: " << img.cols << "x" << img.rows << endl;

      Vector2i img_size(640, 480);
      cout << "ITM Vector2i: " << img_size.x << ", " << img_size.y << endl;

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
//  dynslam::George();
  dynslam::ScaffoldingLoop();
}
