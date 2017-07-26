#include "frame_cpu.h"
#include "../../../feature/feature_depthMap.h"
#include "../../../commDefs.h"

namespace VGUGV
{
  namespace Common 
  { 
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::~Frame_CPU()
    {
       for(int i = 0; i < mnPyramidLevels; i++)
      {
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
      delete [] mpPyramidImages; mpPyramidImages = NULL;
      delete [] mpPyramidImageGradientMag; mpPyramidImageGradientMag = NULL;
      delete [] mpPyramidImageGradientVec; mpPyramidImageGradientVec = NULL;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::computeImagePyramids(int nTotalLevels)
    {   
      if(nTotalLevels < 1) return;
      if(mpPyramidImages != NULL) return;
      mnPyramidLevels = nTotalLevels;

		std::cout << "Computing image pyramids for " << nTotalLevels << " levels..." << std::endl;
      
      // dynamically allocate memory space for mpPyramidImages
      mpPyramidImages = new unsigned char*[nTotalLevels];
      
      // copy first level directly
      mpPyramidImages[0] = new unsigned char[mnRows * mnCols];
      memcpy(mpPyramidImages[0], mpGrayImageData_CPU, mnRows * mnCols);
      
      for (int i = 1; i < nTotalLevels; i++)
      {
        printf("Computing pyramid level %d/%d (1 pre-copied)...\n", (i + 1), nTotalLevels);
	int scale = 1 << i;
	int nRows = mnRows / scale;
	int nCols = mnCols / scale;
	int nSize = nRows * nCols;
	
	mpPyramidImages[i] = new unsigned char[nSize];
	
	unsigned char* p_dst = mpPyramidImages[i];
	unsigned char* p_src = mpPyramidImages[i-1];
	
	for (int r = 0; r < nRows; r++)
	{
	  for (int c = 0; c < nCols; c++, p_dst++)
	  {
	    int rowIndexInPreviousImage = 2 * r + 1;
	    int colIndexInPreviousImage = 2 * c + 1;
	    
	    if (c == nCols - 1 || r == nRows - 1) // on bounary
	    {
	      int index = rowIndexInPreviousImage * nCols * 2 + colIndexInPreviousImage;
	      p_dst[0] = (p_src + index)[0];
	      continue;
	    }
	    
	    int P00_index = (rowIndexInPreviousImage - 1) * nCols * 2 + (colIndexInPreviousImage - 1);
	    int P01_index = P00_index + 2;
	    int P10_index = P00_index + nCols * 4;
	    int P11_index = P10_index + 2;
	    
	    unsigned short sum =  static_cast<unsigned short>((p_src + P00_index)[0]) + 
	    static_cast<unsigned short>((p_src + P01_index)[0]) + 
	    static_cast<unsigned short>((p_src + P10_index)[0]) +
	    static_cast<unsigned short>((p_src + P11_index)[0]);
	    
	    p_dst[0] = static_cast<unsigned char>(sum / 4);
	  }
	}
	{
	  p_dst = NULL;
	  p_src = NULL;
	}
      }
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::computeImagePyramidsGradients(int nTotalLevels)
    {
      if (nTotalLevels == 0) return;
      if (mpPyramidImageGradientMag != NULL) return;
      mnPyramidLevels = nTotalLevels;
      
      if (mpPyramidImages == NULL)
      {
	printf("Frame_CPU::computeImagePyramidsGradients failed due to no pyramid image data ...\n");
	return;
      }

      printf("Pyramid data present. Will compute gradients and their magnitudes...\n");

      mpPyramidImageGradientMag = new float*[nTotalLevels];
      mpPyramidImageGradientVec = new Eigen::Vector2f*[nTotalLevels];
      
      for (int i = 0; i < nTotalLevels; i++)
      {
        printf("Computing gradients for level %d.\n", i+1);
	int scale = 1 << i;
	int nRows = mnRows / scale;
	int nCols = mnCols / scale;
	
	mpPyramidImageGradientMag[i] = new float[nRows * nCols];
	mpPyramidImageGradientVec[i] = new Eigen::Vector2f[nRows * nCols];
	
	unsigned char*   pImageSrc = mpPyramidImages[i];
	float*           pGradientMag = mpPyramidImageGradientMag[i];
	Eigen::Vector2f* pGradientVec = mpPyramidImageGradientVec[i];
	
	for (int r = 0; r < nRows; r++) {
	  for (int c = 0; c < nCols; c++) {
	    int index = r * nCols + c;
	    
	    if (r == 0 || r == nRows - 1 || c == 0 || c == nCols - 1)
	    {
	      pGradientMag[index] = 0.0f;
	      pGradientVec[index] = Eigen::Vector2f(0.0f, 0.0f);
	      continue;
	    }
	    
	    int rInTopLevel = scale * r + scale - 1;
	    int cInTopLevel = scale * c + scale - 1;

        // TODO(andrei): re-enable if we want masking support in DynSLAM.
//	    if (!pixelLieOutsideImageMask(rInTopLevel, cInTopLevel)) // check whether it is being masked or not...
//	    {
//	      pGradientMag[index] = 0.0f;
//	      pGradientVec[index] = Eigen::Vector2f(0.0f, 0.0f);
//	      continue;
//	    }
	    
	    int indexRght = index + 1;
	    int indexLeft = index - 1;
	    int indexUp = index - nCols;
	    int indexBot = index + nCols;
	    
	    float indensityRght = pImageSrc[indexRght];
	    float indensityLeft = pImageSrc[indexLeft];
	    float indensityUp = pImageSrc[indexUp];
	    float indensityBot = pImageSrc[indexBot];
	    
	    float dx = (indensityRght - indensityLeft)*0.5f;
	    float dy = (indensityBot  - indensityUp)*0.5f;
	    
	    pGradientMag[index] = sqrt(dx * dx + dy * dy);
	    pGradientVec[index] = Eigen::Vector2f(dx, dy);
	  }
	} // end for r c 
	pImageSrc = NULL;
	pGradientMag = NULL;
	pGradientVec = NULL;
      } // end for i
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    bool Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::pixelLieOutsideImageMask(int r, int c)
    {
      unsigned char* pMask = mpImageMaskData_CPU;
      if(pMask == NULL) return false;
      
      int index = r*mnCols + c;
      if(index < 0 || pMask[index] < 100) return false;
      return true;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    unsigned char* Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::getGrayImage(DEVICE_TYPE device)
    {
      return mpGrayImageData_CPU;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    unsigned char* Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::getPyramidImage(int nLevel, DEVICE_TYPE type)
    {
      if(mpPyramidImages == NULL) return NULL;
      return mpPyramidImages[nLevel];
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    float* Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::getPyramidImageGradientMag(int nLevel, DEVICE_TYPE type)
    {
      if(mpPyramidImageGradientMag == NULL) return NULL;
      return mpPyramidImageGradientMag[nLevel]; 
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Eigen::Vector2f* Frame_CPU<T_FeatureType, T_FeatureDescriptorType>::getPyramidImageGradientVec(int nLevel, DEVICE_TYPE device)
    {
      if(mpPyramidImageGradientVec == NULL) return NULL;
      return mpPyramidImageGradientVec[nLevel]; 
    }
    
    template class Frame_CPU<Feature_depthMap<DepthHypothesis_GMM>, DepthHypothesis_GMM>;
  }
}