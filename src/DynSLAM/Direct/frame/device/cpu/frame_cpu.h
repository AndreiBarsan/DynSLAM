#ifndef _VGUGV_COMMON_FRAME_CPU_
#define _VGUGV_COMMON_FRAME_CPU_

#include "../../frame.h"
#include <memory>

namespace VGUGV
{
 namespace Common 
 {
   template<class T_FeatureType, class T_FeatureDescriptorType>
   class Frame_CPU : public Frame<T_FeatureType, T_FeatureDescriptorType>
   { 
   public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     typedef std::shared_ptr<Frame_CPU<T_FeatureType, T_FeatureDescriptorType> > Ptr;    
     typedef Frame<T_FeatureType, T_FeatureDescriptorType> Base;
   
   protected:
     using Base::mpCameraModel;
     using Base::mpImageData_CPU;
     using Base::mpGrayImageData_CPU;
     using Base::mpImageMaskData_CPU;
     using Base::mpPyramidImages;
     using Base::mpPyramidImageGradientMag;
     using Base::mpPyramidImageGradientVec;
     using Base::mnRows;
     using Base::mnCols;
     using Base::mnChannels;
     using Base::mnPyramidLevels;
     
   public:
     Frame_CPU(int frameId, const CameraBase::Ptr& camera, const unsigned char* imageData, const unsigned char* maskImage, int nRows, int nCols, int nChannels)
      : Base(frameId, camera, imageData, maskImage, nRows, nCols, nChannels){};
     ~Frame_CPU();
   public:
      void computeImagePyramids(int nTotalLevels);
      void computeImagePyramidsGradients(int nTotalLevels);
      bool pixelLieOutsideImageMask(int r, int c);
      
      unsigned char* getGrayImage(DEVICE_TYPE device = DEVICE_TYPE::CPU);
      size_t         getGrayImageCUDAPitch() { return 0;};
      unsigned char* getPyramidImage(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
      float*         getPyramidImageGradientMag(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
      Eigen::Vector2f* getPyramidImageGradientVec(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
   };
 } 
}
#endif