#include "feature_depthMap.h"
#include "../commDefs.h"

namespace VGUGV
{
  namespace Common
  { 
    template<class T_FeatureDescriptorType> Feature_depthMap<T_FeatureDescriptorType>::Feature_depthMap(int nRows, int nCols)
    : mnRows(nRows)
    , mnCols(nCols)
    , mpDepthMap(NULL)
    , mnFeatureSize(NULL)
    , mnPyramidLevels(0)
    , mpIndexMap4FeatureDownsampling(NULL){}
    
    template<class T_FeatureDescriptorType> Feature_depthMap<T_FeatureDescriptorType>::~Feature_depthMap()
    {
      delete[] mpDepthMap; mpDepthMap = NULL;
      delete[] mnFeatureSize; mnFeatureSize = NULL;
      delete[] mpIndexMap4FeatureDownsampling; mpIndexMap4FeatureDownsampling = NULL;
    }
    
    template<class T_FeatureDescriptorType> 
    void Feature_depthMap<T_FeatureDescriptorType>::copyFeatureDescriptors(T_FeatureDescriptorType* featureDescs, 
									   int nNumOfDescriptors, 
									   int nPyramidLevel,
									   unsigned char** pPyramidImage)
    {
      if(nPyramidLevel < 1) return;      
      mnPyramidLevels = nPyramidLevel;
      // allocate new memory space
      if(mpDepthMap == NULL)
      {
	mpDepthMap = new T_FeatureDescriptorType*[nPyramidLevel];
	for(int i = 0; i < nPyramidLevel; i++) mpDepthMap[i] = NULL;
	mnFeatureSize = new int[nPyramidLevel];
	memset(mnFeatureSize, 0, sizeof(int) * nPyramidLevel);
      }
      // copy top level feature descriptors 
      mnFeatureSize[0] = nNumOfDescriptors; 
      if(mpDepthMap[0] == NULL) mpDepthMap[0] = new T_FeatureDescriptorType[nNumOfDescriptors];	
      memcpy(mpDepthMap[0], featureDescs, sizeof(T_FeatureDescriptorType) * nNumOfDescriptors);
      // downsample feature descriptors if necessary
      if(nPyramidLevel > 1)
      {
	// create index map for the ease of downsampling
	if(mpIndexMap4FeatureDownsampling == NULL)
	{
	  mpIndexMap4FeatureDownsampling = new int*[nPyramidLevel];
	  for(int i = 0; i < nPyramidLevel; i++) mpIndexMap4FeatureDownsampling[i] = NULL;
	  
	  // initialize first level
	  int nTotalEntries = mnRows * mnCols;
	  if(mpIndexMap4FeatureDownsampling[0] == NULL) mpIndexMap4FeatureDownsampling[0] = new int[nTotalEntries];
	  for(int i = 0; i < nTotalEntries; i++) mpIndexMap4FeatureDownsampling[0][i] = -1;	
	  for(int i = 0; i < nNumOfDescriptors; i++)
	  {
	    T_FeatureDescriptorType depth = featureDescs[i];
	    Eigen::Vector2i pixel_rc = depth.pixel;
	    int index = pixel_rc(0) * mnCols + pixel_rc(1);
	    mpIndexMap4FeatureDownsampling[0][index] = i;
	  }
	}
	
	// downsample depth map
	for(int i = 1; i < nPyramidLevel; i++)
	{
	  const unsigned char* pImageData = pPyramidImage[i];
	  int scale = 1 << i;
	  int nRows = mnRows / scale;
	  int nCols = mnCols / scale;
	  if(mpIndexMap4FeatureDownsampling[i] == NULL) mpIndexMap4FeatureDownsampling[i] = new int[nRows * nCols];
	  
	  std::vector<T_FeatureDescriptorType> ValidFeatureVector;
	  for(int r = 0; r < nRows; r++) for(int c = 0; c < nCols; c++)
	  {
	    int index = r * nCols + c;
	    mpIndexMap4FeatureDownsampling[i][index] = -1;
	    
	    int rowIndexInPrevLevel = 2 * r + 1; int colIndexInPrevLevel = 2 * c + 1;
	    int indexInPrevLevel = rowIndexInPrevLevel * nCols * 2 + colIndexInPrevLevel;
	      
	    int featureIndex = mpIndexMap4FeatureDownsampling[i - 1][indexInPrevLevel];
	    if(featureIndex < 0) continue;
	    
	    mpIndexMap4FeatureDownsampling[i][index] = featureIndex;
	    T_FeatureDescriptorType temp = featureDescs[featureIndex];
	    temp.pixel(0) = r; temp.pixel(1) = c;
	    temp.intensity = pImageData[r * nCols + c];
	    ValidFeatureVector.push_back(temp);
	  }
	  // copy to local buffer
	  mnFeatureSize[i] = ValidFeatureVector.size();
	  if(mpDepthMap[i] == NULL) mpDepthMap[i] = new T_FeatureDescriptorType[ValidFeatureVector.size()];
	  memcpy(mpDepthMap[i], ValidFeatureVector.data(), ValidFeatureVector.size() * sizeof(T_FeatureDescriptorType));
	}
      }
    }
    
    template<class T_FeatureDescriptorType> 
    T_FeatureDescriptorType* Feature_depthMap<T_FeatureDescriptorType>::getFeatureDescriptors(int input_pyramidLevel)
    {
      if(input_pyramidLevel + 1 > mnPyramidLevels)
      {
	printf("[Feature_depthMap] fails to retrieve depth map due to uninitialized depth map data... (requested pyramid level %d)\n", input_pyramidLevel);
	return NULL;
      }
      return mpDepthMap[input_pyramidLevel];
    }
    
    template<class T_FeatureDescriptorType> 
    int Feature_depthMap<T_FeatureDescriptorType>::getFeatureSize(int in_pyramidLevel)
    {
      if(in_pyramidLevel + 1 > mnPyramidLevels || mnFeatureSize == NULL)
      {
	printf("[Feature_depthMap] fails to retrieve depth map due to uninitialized depth map data...\n");
	return 0;
      }
      return mnFeatureSize[in_pyramidLevel];
    }
    
    template<class T_FeatureDescriptorType> 
    std::vector<Eigen::Vector3f> Feature_depthMap<T_FeatureDescriptorType>::get3DFeaturePCL()
    {
      if(mpDepthMap == NULL) return std::vector<Eigen::Vector3f>();
      T_FeatureDescriptorType* pFeatureDescs = mpDepthMap[0];
      int nSize = mnFeatureSize[0];
      
      std::vector<Eigen::Vector3f> pcl;
      pcl.reserve(nSize);
      
      for(int i = 0; i < nSize; i++)
      {
	float depth = pFeatureDescs[i].rayDepth;
	Eigen::Vector3f unitRay = pFeatureDescs[i].unitRay;
	pcl.push_back(unitRay * depth);
      }
      return pcl;
    }
    
    template<class T_FeatureDescriptorType> 
    std::vector<Eigen::Matrix<float, 6, 1> > Feature_depthMap<T_FeatureDescriptorType>::get3DFeatureColoredPCL()
    {
      if(mpDepthMap == NULL) return std::vector<Eigen::Matrix<float, 6, 1> >();
      T_FeatureDescriptorType* pFeatureDescs = mpDepthMap[0];
      int nSize = mnFeatureSize[0];
      
      std::vector<Eigen::Matrix<float, 6, 1> > pcl;
      pcl.reserve(nSize);
      
      for(int i = 0; i < nSize; i++)
      {
	float depth = pFeatureDescs[i].rayDepth;
	Eigen::Vector3f unitRay = pFeatureDescs[i].unitRay;
	Eigen::Vector3f texture = pFeatureDescs[i].texture; // rgb color
	Eigen::Matrix<float, 6, 1> point;
	point.block<3,1>(0, 0) = unitRay * depth;
	point.block<3,1>(3, 0) = texture;
	
	pcl.push_back(point);
      }
      return pcl;
    }
    
    template class Feature_depthMap<DepthHypothesis_GMM>;
  }
}