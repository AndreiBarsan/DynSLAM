
#include <iostream>

#include <opencv/cv.h>
#include <highgui.h>

#include "ImageSourceEngine.h"

// TODO(andrei): Configure InfiniTAM's includes so we can be less verbose here.
#include "../InfiniTAM/InfiniTAM/ITMLib/Objects/ITMView.h"
#include "../InfiniTAM/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <backward.hpp>

// TODO(andrei): TODO file for code-specific TODOs.
// TODO(andrei): clang-format.

namespace dynslam {
  using namespace std;
  using namespace ITMLib::Objects;

  enum Action {
    kIdle,
    kProcessFrame,
    kExit
  };

  const unsigned char kEscAsciiCode = 27;

  /// \brief A simple reconstruction GUI based on the InfiniTAM UIEngine.
  class DynSlamGlutGui {
  public:
    // Necessary for the GLUT callbacks.

    /// \brief Singleton wrapper for supporting GLUT callbacks which take raw function pointers.
    static DynSlamGlutGui* Instance() {
      if (instance == nullptr) {
        instance = new DynSlamGlutGui();
      }

      return instance;
    }

    // TODO(andrei): Write own image source, since your requirements are different!
    void Initialize(ITMMainEngine *itm_static_scene_engine_, ImageSourceEngine *image_source) {

      this->image_source_ = image_source;

      window_size_.x = image_source->getDepthImageSize().x;
      window_size_.y = image_source->getDepthImageSize().y;

      this->itm_static_scene_engine_ = itm_static_scene_engine_;
      this->current_frame_no_ = 0;

      int fake_argc = 0;
      glutInit(&fake_argc, nullptr);
      glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
      glutInitWindowSize(window_size_.x, window_size_.y);
      glutCreateWindow("DynSLAM Lightweight GUI");

      // TODO(andrei): Multiple panel support, if you don't go QT.
      glGenTextures(1, panel_texture_ids_);

      glutDisplayFunc(DynSlamGlutGui::GlutDisplay);
      glutKeyboardUpFunc(DynSlamGlutGui::GlutKeyboard);
      glutIdleFunc(DynSlamGlutGui::GlutIdle);

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

      // TODO(andrei): Refactor DynSLAM as separate class, and call it here. DynSLAM should
      // handle stuff like depth, semantics, separate volumetric integration, etc.

      // Forward them to InfiniTAM for the background reconstruction.
      itm_static_scene_engine_->ProcessFrame(input_rgb_image_, input_raw_depth_image_);
      ITMSafeCall(cudaThreadSynchronize());

      current_frame_no_++;
    }

    void Run() {
      glutMainLoop();
    }

    void Shutdown() {
      cout << "DynSLAM GLUT GUI shutting down." << endl;
      // TODO(andrei): Clean up after yourself.
    }

  private:
    DynSlamGlutGui() : needs_refresh_(false), next_action_(Action::kIdle) { }

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

    uint panel_texture_ids_[1];
    bool needs_refresh_;
    Action next_action_;

    /***********************************************************************************************
     * Static methods used as GLUT UI callbacks.
     */
    static DynSlamGlutGui *instance;

    static void GlutDisplay() {
      DynSlamGlutGui *gui = Instance();
      gui->itm_static_scene_engine_->GetImage(
          gui->out_image_,
          ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST);

      glClear(GL_COLOR_BUFFER_BIT);
      glColor3f(1.0f, 1.0f, 1.0f);
      glEnable(GL_TEXTURE_2D);

      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      {
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        {
          float winReg[][4] = {
              {0, 0, 1, 1}
          };
          glEnable(GL_TEXTURE_2D);
          // TODO Maybe loop begin, or delete this.
          int w = 0;
          glBindTexture(GL_TEXTURE_2D, gui->panel_texture_ids_[0]);
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                       gui->out_image_->noDims.x, gui->out_image_->noDims.y,
                       0, GL_RGBA, GL_UNSIGNED_BYTE,
                       gui->out_image_->GetData(MEMORYDEVICE_CPU));
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glBegin(GL_QUADS); {
//            glTexCoord2f(0, 1); glVertex2f(winReg[w][0], winReg[w][1]); // glVertex2f(0, 0);
//            glTexCoord2f(1, 1); glVertex2f(winReg[w][2], winReg[w][1]); // glVertex2f(1, 0);
//            glTexCoord2f(1, 0); glVertex2f(winReg[w][2], winReg[w][3]); // glVertex2f(1, 1);
//            glTexCoord2f(0, 0); glVertex2f(winReg[w][0], winReg[w][3]); // glVertex2f(0, 1);

            glTexCoord2f(0, 1); glVertex2f(0, 0);
            glTexCoord2f(1, 1); glVertex2f(1, 0);
            glTexCoord2f(1, 0); glVertex2f(1, 1);
            glTexCoord2f(0, 0); glVertex2f(0, 1);
          }
          glEnd();

          // Maybe loop end
          glDisable(GL_TEXTURE_2D);
        }
        glPopMatrix();
      }
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();

      // TODO(andrei): Render UI text here, if any.
      glColor3f(1.0f, 0.0f, 0.0f); glRasterPos2f(0.85f, -0.962f);


      // End text UI code.

      cout << "Swapping buffers..." << endl;
      glutSwapBuffers();
      cout << "Done." << endl;
      gui->needs_refresh_ = false;
    }

    static void GlutIdle() {
      DynSlamGlutGui *gui = Instance();

      switch (gui->next_action_) {
        case kIdle:
          break;
        case kProcessFrame:
          gui->ProcessFrame();
          gui->needs_refresh_ = true;
          gui->next_action_ = Action::kIdle;
          break;
        case kExit:
          glutLeaveMainLoop();
          break;
      }

      if (gui->needs_refresh_) {
        glutPostRedisplay();
      }
    }

    static void GlutKeyboard(unsigned char key, int x, int y) {
      DynSlamGlutGui *gui = Instance();

      switch(key) {
        case 'n':
          cout << endl << "Processing frame [" << gui->current_frame_no_ << "]..." << endl;
          gui->next_action_ = Action::kProcessFrame;
          break;

        case 'e':
        case 'q':
        case  kEscAsciiCode:
          cout << endl << "Exiting..." << endl;
          gui->next_action_ = Action::kExit;
          break;

        default:
          cout << "Unbound key up: " << key << ", x = " << x << ", y =" << y << endl;
      }
    }

  };

  DynSlamGlutGui* DynSlamGlutGui::instance = nullptr;

  /// \brief Crude main loop running InfiniTAM.
  /// In the future, this will be refactored into multiple components.
  void ScaffoldingLoop() {
    DynSlamGlutGui *gui = DynSlamGlutGui::Instance();

//    ITMSafeCall(cudaMemcpy(nullptr, nullptr, 42, cudaMemcpyDeviceToHost));

    // TODO pass as arg to initialize
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

    gui->Initialize(static_scene_engine, image_source);
    gui->Run();
    gui->Shutdown();
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
//int main() {
//  dynslam::ScaffoldingLoop();
//  return 0;
//}
