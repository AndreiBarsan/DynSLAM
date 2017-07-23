#ifndef _VGUGV_COMMON_FEATURE_DEPTHMAP_
#define _VGUGV_COMMON_FEATURE_DEPTHMAP_

#include <Eigen/Dense>
#include <memory>

#include "../cameraBase.h"

namespace VGUGV
{
  namespace Common 
  {    
    template<class T_FeatureDescriptorType>
    class Feature_depthMap
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef std::shared_ptr<Feature_depthMap<T_FeatureDescriptorType> > Ptr; 
    
    public:
      Feature_depthMap(int nRows, int nCols);
      ~Feature_depthMap();
      
      //! A normal member function which copies feature descriptors
      /*!
      \param pDescriptors pointer to feature descriptors
      \param nNumOfDescriptors        total number of feature descriptors
      \param nTotalPyramidLevels if not equal to 1, then usually it is for semidense/dense depth map, then nSize = image_height * image_width; For semi-dense map, feature_descriptor should have flag to indicate whether current depth is valid; It does not matter if pyramid level is 1;
      */      
      void copyFeatureDescriptors(T_FeatureDescriptorType* featureDescs, int nNumOfDescriptors, int nPyramidLevel = 0, unsigned char** pPyramidImage = NULL);    
      
      T_FeatureDescriptorType* getFeatureDescriptors(int input_pyramidLevel = 0);
      int  getFeatureSize(int in_pyramidLevel = 0);
      
      std::vector<Eigen::Vector3f> get3DFeaturePCL();
      std::vector<Eigen::Matrix<float, 6, 1> > get3DFeatureColoredPCL();
       
    private:
      T_FeatureDescriptorType** mpDepthMap; 
      int* mnFeatureSize;
      int** mpIndexMap4FeatureDownsampling; 
      int mnRows, mnCols;
      int mnPyramidLevels;
    };
  }
}

#endif