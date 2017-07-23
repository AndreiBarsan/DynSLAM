#include "multiFrames.h"
#include "common/feature/feature_depthMap.h"
#include "common/commDefs.h"

namespace VGUGV {
  namespace Common{
    template<class T_FeatureType, class T_FeatureDescriptorType>
    MultiFrames<T_FeatureType, T_FeatureDescriptorType>::MultiFrames(int seqID, int nFrames)
    : mMultiFrameID(seqID)
    , mnFrames(nFrames)
    {
      mvFrames.reserve(mnFrames);
    };
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void MultiFrames<T_FeatureType, T_FeatureDescriptorType>::insertNewFrame(const BaseFramePtr& frame)
    {
      mvFrames.push_back(frame);
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    typename MultiFrames<T_FeatureType, T_FeatureDescriptorType>::BaseFramePtr MultiFrames<T_FeatureType, T_FeatureDescriptorType>::getFrame(int index)
    {
      return mvFrames.at(index); 
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    void MultiFrames<T_FeatureType, T_FeatureDescriptorType>::setMultiFramePose(Transformation multiFramePose)
    {
      mMultiFramePose = multiFramePose;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>
    Transformation MultiFrames<T_FeatureType, T_FeatureDescriptorType>::getMultiFramePose()
    {
      return mMultiFramePose;
    }
    
    template<class T_FeatureType, class T_FeatureDescriptorType>      
    int MultiFrames<T_FeatureType, T_FeatureDescriptorType>::getMultiFrameID()
    {
      return mMultiFrameID;
    }
    
    template class MultiFrames<Feature_depthMap<DepthHypothesis_GMM>, DepthHypothesis_GMM>;
  }
}
