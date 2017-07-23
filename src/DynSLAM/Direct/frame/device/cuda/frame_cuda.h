#ifndef _VGUGV_COMMON_FRAME_CUDA_
#define _VGUGV_COMMON_FRAME_CUDA_

#include "../../frame.h"
#include <memory>
#include "../../../commDefs.h"
#include "../../../cudaDefs.h"

namespace VGUGV
{
 namespace Common
 {
   template<class T_FeatureType, class T_FeatureDescriptorType>
   class Frame_CUDA : public Frame<T_FeatureType, T_FeatureDescriptorType>
   { 
   public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     typedef std::shared_ptr<Frame_CUDA<T_FeatureType, T_FeatureDescriptorType> > Ptr;    
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
     using Base::mFrameID;
     using Base::mnPyramidLevels;
     
   public:
     Frame_CUDA(int frameId, const CameraBase::Ptr& camera, const unsigned char* imageData, const unsigned char* maskImage, int nRows, int nCols, int nChannels);
     ~Frame_CUDA();
   
   public:
      void computeImagePyramids(int nTotalLevels);
      void computeImagePyramidsGradients(int nTotalLevels);
      bool pixelLieOutsideImageMask(int r, int c);
      
      unsigned char* getGrayImage(DEVICE_TYPE device = DEVICE_TYPE::CPU);
      size_t         getGrayImageCUDAPitch();
      unsigned char* getPyramidImage(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
      float*         getPyramidImageGradientMag(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
      Eigen::Vector2f* getPyramidImageGradientVec(int level, DEVICE_TYPE device = DEVICE_TYPE::CPU);
      
   private:
     CUDA_PitchMemory<unsigned char> mpImageData_CUDA;
     CUDA_PitchMemory<unsigned char> mpImageMaskData_CUDA;
     CUDA_PitchMemory<unsigned char>* mpPyramidImages_CUDA;
     CUDA_PitchMemory<float>* mpPyramidImageGradientMag_CUDA;
     CUDA_PitchMemory<Eigen::Vector2f>* mpPyramidImageGradientVec_CUDA;
     
     cudaChannelFormatDesc mFrameCuda_uchar1ChannelDesc;
   };
 } 
}

#endif