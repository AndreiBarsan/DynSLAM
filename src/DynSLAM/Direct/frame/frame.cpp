#include "frame.h"
#include "../feature/feature_depthMap.h"
#include "../commDefs.h"

namespace VGUGV
{
  namespace Common
  {
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Frame<T_FeatureType, T_FeatureDescriptorType>::Frame(int frameId, const CameraBase::Ptr& camera, const unsigned char* imageData, const unsigned char* maskImageData, int nRows, int nCols, int nChannels) 
    : mFrameID(frameId)
    , mpCameraModel(camera)
    , mnRows(nRows)
    , mnCols(nCols)
    , mnChannels(nChannels)
    , mpImageMaskData_CPU(NULL)
    , mpPyramidImages(NULL)
    , mpDepthMap_CPU(NULL)
    , mpPyramidImageGradientMag(NULL)
    , mpPyramidImageGradientVec(NULL)
    , mFeatureInstance(T_FeatureType(mnRows, mnCols))
    , mnPyramidLevels(0)
    {
      int nDataSizeInBytes = nRows * nCols * nChannels;
      mpImageData_CPU = new unsigned char[nDataSizeInBytes];
      memcpy(mpImageData_CPU, imageData, nDataSizeInBytes); 
      
      if(nChannels == 1)
      {
	mpGrayImageData_CPU = new unsigned char[nRows * nCols];
	memcpy(mpGrayImageData_CPU, imageData, nRows * nCols);
      }
      else if(nChannels == 3)
      {
	mpGrayImageData_CPU = new unsigned char[nRows * nCols];
	for(int r = 0; r < nRows; r ++)
	{
	 for(int c = 0; c < nCols; c++)
	 {
	   int index = r * nCols + c;
	   int index3 = index * 3;
	   mpGrayImageData_CPU[index] = 0.299 * imageData[index3] + 0.587 * imageData[index3 + 1] + 0.114 * imageData[index3 + 2];
	 }
	}
      }
      
      if(maskImageData != NULL)
      {
	mpImageMaskData_CPU = new unsigned char[nRows * nCols];
	memcpy(mpImageMaskData_CPU, maskImageData, nRows * nCols);
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Frame<T_FeatureType, T_FeatureDescriptorType>::~Frame()
    {
      delete mpImageData_CPU; mpImageData_CPU = NULL;
      delete mpGrayImageData_CPU; mpGrayImageData_CPU = NULL;
      delete mpImageMaskData_CPU; mpImageMaskData_CPU = NULL;
      delete mpDepthMap_CPU; mpDepthMap_CPU = NULL;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    int Frame<T_FeatureType, T_FeatureDescriptorType>::getFrameID()
    {
      return mFrameID;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Eigen::Vector3f Frame<T_FeatureType, T_FeatureDescriptorType>::getImageRGBTexture(int r, int c)
    {
      int index = r * mnCols + c;
      Eigen::Vector3f texture;
      switch(mnChannels)
      {
	case 1:
	  texture(0) = mpImageData_CPU[index];
	  texture(1) = texture(0);
	  texture(2) = texture(1);
	  break;
	case 3:  // RGB
	  int index3 = 3 * index;
	  texture(0) = mpImageData_CPU[index3];
	  texture(1) = mpImageData_CPU[index3 + 1];
	  texture(2) = mpImageData_CPU[index3 + 2];
	  break;
      }
      return texture;
    }
      
    template<class T_FeatureType, class T_FeatureDescriptorType>
    CameraBase::Ptr& Frame<T_FeatureType, T_FeatureDescriptorType>::getCameraModel()
    {
      return mpCameraModel;
    }
    
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Eigen::Vector2i Frame<T_FeatureType, T_FeatureDescriptorType>::getFrameSize()
    {
      Eigen::Vector2i size(mnRows, mnCols);
      return size;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    unsigned char* Frame<T_FeatureType, T_FeatureDescriptorType>::getRawImageData()
    {
      return mpImageData_CPU;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame<T_FeatureType, T_FeatureDescriptorType>::copyDepthMapData(float* pDepthMap)
    {
      mpDepthMap_CPU = new float[mnRows * mnCols];
      memcpy(mpDepthMap_CPU, pDepthMap, mnRows * mnCols * sizeof(float));
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    float* Frame<T_FeatureType, T_FeatureDescriptorType>::getDepthMapData()
    {
      return mpDepthMap_CPU;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame<T_FeatureType, T_FeatureDescriptorType>::copyFeatureDescriptors(T_FeatureDescriptorType* descriptors, int nSize, int nTotalPyramidLevels)
    {
      mFeatureInstance.copyFeatureDescriptors(descriptors, nSize, nTotalPyramidLevels, mpPyramidImages);
    }   
    
    template<class T_FeatureType, class T_FeatureDescriptorType>    
    T_FeatureDescriptorType* Frame<T_FeatureType, T_FeatureDescriptorType>::getFeatureDescriptors(int input_pyramidLevel)
    {
      return mFeatureInstance.getFeatureDescriptors(input_pyramidLevel);
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>    
    int  Frame<T_FeatureType, T_FeatureDescriptorType>::getFeatureSize(int in_pyramidLevel)
    {
      return mFeatureInstance.getFeatureSize(in_pyramidLevel);
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>    
    std::vector<Eigen::Vector3f> Frame<T_FeatureType, T_FeatureDescriptorType>::get3DFeaturePCL()
    {
      return mFeatureInstance.get3DFeaturePCL();
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType> 
    std::vector<Eigen::Matrix<float, 6, 1> > Frame<T_FeatureType, T_FeatureDescriptorType>::get3DFeatureColoredPCL()
    {
      return mFeatureInstance.get3DFeatureColoredPCL();
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>    
    std::vector<Eigen::Vector2i> Frame<T_FeatureType, T_FeatureDescriptorType>::samplePixels(int nTotal)
    {
      if(mpPyramidImageGradientMag == NULL) return std::vector<Eigen::Vector2i>();
      float* pGradientMap = mpPyramidImageGradientMag[0];
      
      // divide image into 32 x 32 blocks
      int nBlockX = mnCols / 32;
      int nBlockY = mnRows / 32;
      int nBlocks = nBlockX * nBlockY;
      
      int gradientHistogram[256]; memset(gradientHistogram, 0, sizeof(int) * 256);
      int* blockThs = new int[nBlocks]; 
      
      for(int r0 = 0; r0 < nBlockY; r0++) for(int c0 = 0; c0 < nBlockX; c0++)
      {
	memset(gradientHistogram, 0, sizeof(int) * 256);
	for(int r1 = 0; r1 < 32; r1++) for(int c1 = 0; c1 < 32; c1++)
	{
	  int r = r0 * 32 + r1;
	  int c = c0 * 32 + c1;
	  if(r > mnRows - 1 || c > mnCols - 1) continue;
	  int index = r * mnCols + c; 
	  int gradient = pGradientMap[index];
	  gradientHistogram[gradient]++;
	}
	
	int nTotalPixelsPerBlk = (1.0 - nTotal / (mnRows * mnCols * 1.0f)) * 1024;
	for(int i = 0; i < 256; i++)
	{
	  nTotalPixelsPerBlk -= gradientHistogram[i];
	  if(nTotalPixelsPerBlk < 0)
	  {
	    blockThs[r0 * nBlockX + c0] = i + 20;
	    break;
	  }
	  blockThs[r0 * nBlockX + c0] = 20;
	}
      }
      // sample pixels
      std::vector<Eigen::Vector2i> sampledPixels; // (r, c)
      
      bool* bSelectedMap = new bool[mnRows * mnCols];
      memset(bSelectedMap, 0, sizeof(bool) * mnRows * mnCols);
      for(int i = 0; i < 4; i++)
      {
	float scale = 1 << i;
	for(int r = mnRows - 1; r > 0; r--) for(int c = mnCols - 1; c > 0; c--)
	{
	  int index = r * mnCols + c;
	  int block_x = c / 32;
	  int block_y = r / 32;
	  int block_th = blockThs[block_y * nBlockX + block_x] / scale;
	  if(!bSelectedMap[index] && pGradientMap[index] < block_th) continue;
	  sampledPixels.push_back(Eigen::Vector2i(r, c));
	  bSelectedMap[index] = true;
	}
	if(sampledPixels.size() > nTotal) break;
      } 
      // clean up heap memory
      delete[] blockThs; blockThs = NULL;
      delete[] bSelectedMap; bSelectedMap = NULL;
      // output pixels
      if(sampledPixels.size() > nTotal) sampledPixels.resize(nTotal);
      return sampledPixels;
    }
    
    template class Frame<Feature_depthMap<DepthHypothesis_GMM>, DepthHypothesis_GMM>;
  }
}