#ifndef _VGUGV_COMMON_FRAME_
#define _VGUGV_COMMON_FRAME_

#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include "../cameraBase.h"
#include "../commDefs.h"
#include "../math/Vector.h"

namespace VGUGV
{
  namespace Common
  {
    template<class T_FeatureType, class T_FeatureDescriptorType>
    class Frame
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef std::shared_ptr<Frame<T_FeatureType, T_FeatureDescriptorType> > Ptr;      
      
    public:
      //! Constructor for base Frame class
      /*!
      \param camera camera model, i.e., pinhole camera model or unified camera model
      \param imageData raw image data 
      \param nRows height of image
      \param nCols width of image
      \param nChannels number of image channels, i.e., 1 --> grayImage, 3 --> color image in RGB encoding
      */
      Frame(int frameID, const CameraBase::Ptr& camera, const unsigned char* imageData, const unsigned char* maskImageData, int nRows, int nCols, int nChannels);
      ~Frame();  
      
    public:
      virtual void computeImagePyramids(int nTotalLevels) = 0;
      virtual void computeImagePyramidsGradients(int nTotalLevels) = 0;
      virtual bool pixelLieOutsideImageMask(int r, int c) = 0;
      
      virtual unsigned char* getGrayImage(DEVICE_TYPE device = DEVICE_TYPE::CPU) = 0;
      virtual size_t         getGrayImageCUDAPitch() = 0;
      virtual unsigned char* getPyramidImage(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU) = 0;      
      virtual float*         getPyramidImageGradientMag(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU) = 0;
      //! A virtual member function taking the pyramid level number and returning the gradient vector map for all pixels.
      /*!
      \param level an integer argument specifying the image pyramid level number.
      \return The pointer to the gradient vector map (Dx, Dy) for current pyramid image.
      */
      virtual Eigen::Vector2f* getPyramidImageGradientVec(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU) = 0; 
            
    public:
      int getFrameID();

      Eigen::Vector3f getImageRGBTexture(int r, int c);
      
      CameraBase::Ptr& getCameraModel();
      
      /*!
      \return 2-int sized vector (nRows, nCols) of frame size.
      */    
      Eigen::Vector2i  getFrameSize();  
      
      //! Get raw image data, which should be 3-channel image on CPU
      unsigned char*   getRawImageData(); // RGB
      
      void   copyDepthMapData(float* pDepthMap);
      float* getDepthMapData();
      
      //! A normal member function which copies feature descriptors
      /*!
      \param pDescriptors pointer to feature descriptors
      \param nSize        total number of feature descriptors
      \param nTotalPyramidLevels if not equal to 1, then usually it is for semidense/dense depth map, then nSize = image_height * image_width; For semi-dense map, feature_descriptor should have flag to indicate whether current depth is valid; It does not matter if pyramid level is 1;
      */      
      void copyFeatureDescriptors(T_FeatureDescriptorType* pDescriptors, int nNumOfDescriptors, int nTotalPyramidLevels = 1);
      
      T_FeatureDescriptorType* getFeatureDescriptors(int input_pyramidLevel = 0);
      
      int  getFeatureSize(int input_pyramidLevel = 0);
      
      
      std::vector<Eigen::Vector3f> get3DFeaturePCL();
      
      std::vector<Eigen::Matrix<float, 6, 1> > get3DFeatureColoredPCL();
      
      // sample at least n high gradient pixels uniformly
      std::vector<Eigen::Vector2i> samplePixels(int nTotal);
      
    protected: 
      CameraBase::Ptr   mpCameraModel;
      float*            mpDepthMap_CPU;
      unsigned char*    mpImageData_CPU;
      unsigned char*    mpGrayImageData_CPU;
      unsigned char*    mpImageMaskData_CPU;
      unsigned char**   mpPyramidImages;
      float**           mpPyramidImageGradientMag;
      Eigen::Vector2f** mpPyramidImageGradientVec;
      int mnRows;
      int mnCols;
      int mnChannels;
      int mFrameID;
      int mnPyramidLevels;
      
      /*
       * Feature related variables
       */
      T_FeatureType    mFeatureInstance;
    };
  }
}

#endif