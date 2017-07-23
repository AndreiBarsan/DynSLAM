#include "frame_cuda.h"
#include "common/feature/feature_depthMap.h"
#include "common/commDefs.h"
#include "common/cudaDefs.h"
#include "common/helperFunctions.hpp"

namespace VGUGV 
{
  namespace Common
  {
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> FrameCuda_GrayImageTexture;
    texture<uchar, cudaTextureType2D, cudaReadModeElementType> FrameCuda_MaskImageTexture;
    
    __device__ bool pixelLieOutsideImageMask(CUDA_PitchMemory<uchar> maskImage, int x, int y);
    __global__ void kernel_imagePyramid(uchar* pTargetImage, size_t pitch, uint step);
    __global__ void kernel_imageGradient(CUDA_PitchMemory<unsigned char> maskImage,
					 CUDA_PitchMemory<float> gradientMagMap,
					 CUDA_PitchMemory<Eigen::Vector2f> gradientVecMap,
					 int nCols,
					 int nRows,
					 size_t scale);
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::Frame_CUDA(int frameId, const CameraBase::Ptr& camera, const unsigned char* imageData, const unsigned char* maskImage, int nRows, int nCols, int nChannels)
      : Base(frameId, camera, imageData, maskImage, nRows, nCols, nChannels) 
      , mpPyramidImages_CUDA(NULL)
      , mpPyramidImageGradientMag_CUDA(NULL)
      , mpPyramidImageGradientVec_CUDA(NULL)
      {
	// always copy gray image to cuda device at the moment	
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&mpImageData_CUDA.dataPtr, &mpImageData_CUDA.pitch, mnCols, mnRows));
	CUDA_SAFE_CALL(cudaMemcpy2D(mpImageData_CUDA.dataPtr, mpImageData_CUDA.pitch, mpGrayImageData_CPU, mnCols, mnCols, mnRows, cudaMemcpyHostToDevice));	
		
	if(maskImage != NULL)
	{
	  CUDA_SAFE_CALL(cudaMallocPitch((void**)&mpImageMaskData_CUDA.dataPtr, &mpImageMaskData_CUDA.pitch, mnCols, mnRows));
	  CUDA_SAFE_CALL(cudaMemcpy2D(mpImageMaskData_CUDA.dataPtr, mpImageMaskData_CUDA.pitch, maskImage, mnCols, mnCols, mnRows, cudaMemcpyHostToDevice));	
	}
	// initiate texture
	FrameCuda_GrayImageTexture.addressMode[0] = cudaAddressModeWrap;
	FrameCuda_GrayImageTexture.addressMode[1] = cudaAddressModeWrap;
	FrameCuda_GrayImageTexture.filterMode = cudaFilterModePoint;
	FrameCuda_GrayImageTexture.normalized = false;
	
	FrameCuda_MaskImageTexture.addressMode[0] = cudaAddressModeWrap;
	FrameCuda_MaskImageTexture.addressMode[1] = cudaAddressModeWrap;
	FrameCuda_MaskImageTexture.filterMode = cudaFilterModePoint;
	FrameCuda_MaskImageTexture.normalized = false;
	
	mFrameCuda_uchar1ChannelDesc = cudaCreateChannelDesc<uchar>();
      }
      
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::~Frame_CUDA()
    {
      CUDA_SAFE_CALL(cudaFree(mpImageData_CUDA.dataPtr));
      CUDA_SAFE_CALL(cudaFree(mpImageMaskData_CUDA.dataPtr));
      for(int i = 0; i < mnPyramidLevels; i++)
      {
	if(mpPyramidImages_CUDA != NULL) CUDA_SAFE_CALL(cudaFree(mpPyramidImages_CUDA[i].dataPtr));
	if(mpPyramidImageGradientMag_CUDA != NULL) CUDA_SAFE_CALL(cudaFree(mpPyramidImageGradientMag_CUDA[i].dataPtr));
	if(mpPyramidImageGradientVec_CUDA != NULL) CUDA_SAFE_CALL(cudaFree(mpPyramidImageGradientVec_CUDA[i].dataPtr));
        
	if(mpPyramidImages != NULL)
	{
	  delete [] mpPyramidImages[i];
	  mpPyramidImages[i] = NULL;
	}
	if(mpPyramidImageGradientMag != NULL)
	{
	  delete [] mpPyramidImageGradientMag[i];
	  mpPyramidImageGradientMag[i] = NULL;
	}
	if(mpPyramidImageGradientVec != NULL)
	{ 
	  delete [] mpPyramidImageGradientVec[i];
	  mpPyramidImageGradientVec[i] = NULL;
	}
      }
      delete [] mpPyramidImages_CUDA; mpPyramidImages_CUDA = NULL;
      delete [] mpPyramidImageGradientMag_CUDA; mpPyramidImageGradientMag_CUDA = NULL;
      delete [] mpPyramidImageGradientVec_CUDA; mpPyramidImageGradientVec_CUDA = NULL;
      delete [] mpPyramidImages; mpPyramidImages = NULL;
      delete [] mpPyramidImageGradientMag; mpPyramidImageGradientMag = NULL;
      delete [] mpPyramidImageGradientVec; mpPyramidImageGradientVec = NULL;
    }
      
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::computeImagePyramids(int nTotalLevels)
    {
      if(nTotalLevels < 1) return;
      if(mpPyramidImages_CUDA != NULL) return;
      
      mnPyramidLevels = nTotalLevels;      
      mpPyramidImages_CUDA = new CUDA_PitchMemory<unsigned char>[nTotalLevels];
      CUDA_SAFE_CALL(cudaMallocPitch(&mpPyramidImages_CUDA[0].dataPtr, &mpPyramidImages_CUDA[0].pitch, mnCols, mnRows));
      CUDA_SAFE_CALL(cudaMemcpy2D(mpPyramidImages_CUDA[0].dataPtr, mpPyramidImages_CUDA[0].pitch, mpImageData_CUDA.dataPtr, mpImageData_CUDA.pitch, mnCols, mnRows, cudaMemcpyDeviceToDevice));
      
      for(int i = 1; i < nTotalLevels; i++)
      {
	int scale = 1 << i;
	int nRows = mnRows / scale;
	int nCols = mnCols / scale;
	
	// bind texture with lower level imageData
	CUDA_SAFE_CALL( cudaBindTexture2D(0, 
					  FrameCuda_GrayImageTexture, 
					  mpPyramidImages_CUDA[i-1].dataPtr, 
				          mFrameCuda_uchar1ChannelDesc, 
				          nCols * 2, 
					  nRows * 2, 
				          mpPyramidImages_CUDA[i-1].pitch));
	CUDA_SAFE_CALL(cudaMallocPitch(&mpPyramidImages_CUDA[i].dataPtr, &mpPyramidImages_CUDA[i].pitch, nCols, nRows));

	// invoke cuda kernel
	int step = 4;
	int thread_x = 8;
	int thread_y = 8;
	dim3 blockDim(thread_x, thread_y); 
	dim3 gridDim(intergerDivUp(nCols, (step * thread_x)), intergerDivUp(nRows, thread_y));
	kernel_imagePyramid<<<gridDim, blockDim>>>(mpPyramidImages_CUDA[i].dataPtr, mpPyramidImages_CUDA[i].pitch, step);
	// unbind texture
	CUDA_SAFE_CALL(cudaUnbindTexture(FrameCuda_GrayImageTexture));
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>:: computeImagePyramidsGradients(int nTotalLevels)
    {
      if(nTotalLevels < 1) return;
      if(mpPyramidImageGradientMag_CUDA != NULL) return;
      
      mnPyramidLevels = nTotalLevels;   
      mpPyramidImageGradientMag_CUDA = new CUDA_PitchMemory<float>[nTotalLevels];
      mpPyramidImageGradientVec_CUDA = new CUDA_PitchMemory<Eigen::Vector2f>[nTotalLevels];
      
      if(mpImageMaskData_CUDA.dataPtr != NULL)
	{
	  CUDA_SAFE_CALL(cudaBindTexture2D(0, 
					   FrameCuda_MaskImageTexture, 
				     mpImageMaskData_CUDA.dataPtr, 
				     mFrameCuda_uchar1ChannelDesc, 
				     mnCols, 
				     mnRows, 
				     mpImageMaskData_CUDA.pitch));
	}
	
      for(int i = 0; i < nTotalLevels; i++)
      {
	int scale = 1 << i;
	int nRows = mnRows / scale;
	int nCols = mnCols / scale;
	
	// bind texture with lower level imageData	
	CUDA_SAFE_CALL( cudaBindTexture2D(0, 
					  FrameCuda_GrayImageTexture, 
					  mpPyramidImages_CUDA[i].dataPtr, 
				          mFrameCuda_uchar1ChannelDesc, 
				          nCols, 
					  nRows, 
				          mpPyramidImages_CUDA[i].pitch));
	CUDA_SAFE_CALL( cudaMallocPitch(&mpPyramidImageGradientMag_CUDA[i].dataPtr, &mpPyramidImageGradientMag_CUDA[i].pitch, nCols * sizeof(float), nRows));
	CUDA_SAFE_CALL( cudaMallocPitch(&mpPyramidImageGradientVec_CUDA[i].dataPtr, &mpPyramidImageGradientVec_CUDA[i].pitch, nCols * sizeof(Eigen::Vector2f), nRows));
	
	// invoke kernel
	int thread_x = 8;
	int thread_y = 8;
	dim3 blockDim(thread_x, thread_y);
	dim3 gridDim(intergerDivUp(nCols, thread_x), intergerDivUp(nRows, thread_y));
	kernel_imageGradient<<<gridDim, blockDim>>>(mpImageMaskData_CUDA, mpPyramidImageGradientMag_CUDA[i], mpPyramidImageGradientVec_CUDA[i], nCols, nRows, 1 << i);
	// unbind texture
	CUDA_SAFE_CALL(cudaUnbindTexture(FrameCuda_GrayImageTexture));
      }
      	CUDA_SAFE_CALL(cudaUnbindTexture(FrameCuda_MaskImageTexture));
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    bool Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::pixelLieOutsideImageMask(int r, int c)
    {
      if(mpImageMaskData_CPU == NULL) return true;
      int index = r * mnCols + c;
      return (mpImageMaskData_CPU[index] > 100);
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    unsigned char* Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::getGrayImage(DEVICE_TYPE device)
    {
      if(device == DEVICE_TYPE::CPU)
      {
	return mpGrayImageData_CPU;
      }
      else
      {
	return mpImageData_CUDA.dataPtr;
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    size_t Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::getGrayImageCUDAPitch()
    {
      return mpImageData_CUDA.pitch;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    unsigned char* Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::getPyramidImage(int level, DEVICE_TYPE device)
    {      
      if(device == DEVICE_TYPE::CPU)
      {
	// download data from GPU
	if(mpPyramidImages == NULL)
	{
	  mpPyramidImages = new unsigned char*[mnPyramidLevels];
	  for (int i = 0; i < mnPyramidLevels; i++)
	  {
	    int scale = 1 << i;
	    int nRows = mnRows / scale;
	    int nCols = mnCols / scale;
	    int nSize = nRows * nCols;
	    mpPyramidImages[i] = new unsigned char[nSize];
	    
	    CUDA_SAFE_CALL(cudaMemcpy2D(mpPyramidImages[i], 
					nCols,
				 mpPyramidImages_CUDA[i].dataPtr,
				 mpPyramidImages_CUDA[i].pitch,
				 nCols,
				 nRows,
				 cudaMemcpyDeviceToHost));
	  }
	}
	return mpPyramidImages[level];
      }
      else
      {
	return mpPyramidImages_CUDA[level].dataPtr;
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    float* Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::getPyramidImageGradientMag(int level, DEVICE_TYPE device)
    {      
      if(device == DEVICE_TYPE::CPU)
      {
	// download data from GPU
	if(mpPyramidImageGradientMag == NULL)
	{
	  mpPyramidImageGradientMag = new float*[mnPyramidLevels];
		    
	  for (int i = 0; i < mnPyramidLevels; i++)
	  {
	    int scale = 1 << i;
	    int nRows = mnRows / scale;
	    int nCols = mnCols / scale;
	    int nSize = nRows * nCols;
	    mpPyramidImageGradientMag[i] = new float[nSize];
	    
	    CUDA_SAFE_CALL(cudaMemcpy2D(mpPyramidImageGradientMag[i], 
					nCols * sizeof(float),
				      mpPyramidImageGradientMag_CUDA[i].dataPtr, 
				 mpPyramidImageGradientMag_CUDA[i].pitch,
			         nCols * sizeof(float),
					nRows,
				 cudaMemcpyDeviceToHost));
	  }
	}
	return mpPyramidImageGradientMag[level];
      }
      else
      {
	return mpPyramidImageGradientMag_CUDA[level].dataPtr;
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Eigen::Vector2f* Frame_CUDA<T_FeatureType, T_FeatureDescriptorType>::getPyramidImageGradientVec(int nLevel, DEVICE_TYPE device)
    {
      if(device == DEVICE_TYPE::CPU)
      {
	// download data from GPU
	if(mpPyramidImageGradientVec == NULL)
	{
	  mpPyramidImageGradientVec = new Eigen::Vector2f*[mnPyramidLevels];
		    
	  for (int i = 0; i < mnPyramidLevels; i++)
	  {
	    int scale = 1 << i;
	    int nRows = mnRows / scale;
	    int nCols = mnCols / scale;
	    int nSize = nRows * nCols;
	    mpPyramidImageGradientVec[i] = new Eigen::Vector2f[nSize];
	    
	    CUDA_SAFE_CALL(cudaMemcpy2D(mpPyramidImageGradientVec[i], 
				      nCols * sizeof(Eigen::Vector2f),
				      mpPyramidImageGradientVec_CUDA[i].dataPtr,
			              mpPyramidImageGradientVec_CUDA[i].pitch,
   			              nCols * sizeof(Eigen::Vector2f),
				      nRows,
				      cudaMemcpyDeviceToHost));
	  }
	}
	return mpPyramidImageGradientVec[nLevel];
      }
      else
      {
	return mpPyramidImageGradientVec_CUDA[nLevel].dataPtr; 
      }   
    }
    
    /* institiate template class */
    template class Frame_CUDA<Feature_depthMap<DepthHypothesis_GMM>, DepthHypothesis_GMM>;
    
    /*****************************************************************************************************************
     * **************************************** Implement CUDA kernels ***********************************************
     * **************************************************************************************************************/  
    __device__ bool pixelLieOutsideImageMask(CUDA_PitchMemory<uchar> maskImage, int x, int y)
    {
      if(maskImage.dataPtr == NULL) return true;
      float intensity = tex2D(FrameCuda_MaskImageTexture, x, y);
      if(intensity < 100) return false;
      return true;
    }
    
    __global__ void kernel_imagePyramid(uchar* pTargetImage, size_t pitch, uint step)
    {
      // step <= 8;
      uint x = (blockIdx.x * blockDim.x + threadIdx.x) * step;
      uint y = blockIdx.y * blockDim.y + threadIdx.y;
      
      // compute pixel coordinate at lower level
      uint y0 = 2 * y + 1;
      uint x0 = 2 * x + 1;
      
      // compute pixel coordinate surrounding (r0, c0)
      uint2 p00 = make_uint2(x0 - 1, y0 - 1);
      uint2 p01 = make_uint2(x0 - 1, y0 + 1);
      
      float I00 = tex2D(FrameCuda_GrayImageTexture, p00.x, p00.y);
      float I01 = tex2D(FrameCuda_GrayImageTexture, p01.x, p01.y);
      
      uchar targetVal4[8];
      for(int i = 0; i < step; ++i)
      {
	uint2 p10 = make_uint2(p00.x + 2, p00.y);
	uint2 p11 = make_uint2(p01.x + 2, p01.y);
	
	float I10 = tex2D(FrameCuda_GrayImageTexture, p10.x, p10.y);
	float I11 = tex2D(FrameCuda_GrayImageTexture, p11.x, p11.y);
		
	targetVal4[i] = (uchar)((I00 + I01 + I10 + I11) * 0.25f);
	
	I00 = I10; I01 = I11;
	p00 = p10; p01 = p11;
      }
      // write to target image 
      memcpy(pTargetImage + y * pitch + x, targetVal4, step);
    }
    
    __global__ void kernel_imageGradient(CUDA_PitchMemory<uchar> maskImage,
					 CUDA_PitchMemory<float> gradientMagMap,
					 CUDA_PitchMemory<Eigen::Vector2f> gradientVecMap,
					 int nCols,
					 int nRows,
					 size_t scale)
    {
      uint x = blockIdx.x * blockDim.x + threadIdx.x;
      uint y = blockIdx.y * blockDim.y + threadIdx.y;
      
      int xInTopLevel = scale * x + scale - 1;
      int yInTopLevel = scale * y + scale - 1;
	    
      float* pGradientMag = (float*)( (char*)gradientMagMap.dataPtr + y * gradientMagMap.pitch) + x;
      Eigen::Vector2f* pGradientVec = (Eigen::Vector2f*)( (char*)gradientVecMap.dataPtr + y * gradientVecMap.pitch) + x;
      
      if(x == 0 || y == 0 || x == nCols - 1 || y == nRows - 1 || !pixelLieOutsideImageMask(maskImage, xInTopLevel, yInTopLevel))
      {
	pGradientMag[0] = 0;
	pGradientVec[0] = Eigen::Vector2f(0, 0);
	return;
      }
      
      // fectch 4 neighbour pixels
      float top =  tex2D(FrameCuda_GrayImageTexture, x,     y - 1);
      float left = tex2D(FrameCuda_GrayImageTexture, x - 1, y);
      float rght = tex2D(FrameCuda_GrayImageTexture, x + 1, y);
      float bot =  tex2D(FrameCuda_GrayImageTexture, x,     y + 1);
      
      float dx = (rght - left) * 0.5f;
      float dy = (bot - top) * 0.5f;
      
      float mag = sqrt(dx * dx + dy * dy);
      pGradientMag[0] = mag;
      pGradientVec[0] = Eigen::Vector2f(dx, dy);
    }
  }
}