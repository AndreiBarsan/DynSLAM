#ifndef _VGUGV_COMMON_MULTIFRAMES_
#define _VGUGV_COMMON_MULTIFRAMES_

#include <memory>
#include <vector>
#include "../frame/frame.h"
#include "../transformation/transformation.h"

namespace VGUGV
{
  namespace Common
  {
    template<class T_FeatureType, class T_FeatureDescriptorType>
    class MultiFrames
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
      typedef std::shared_ptr<MultiFrames<T_FeatureType, T_FeatureDescriptorType> > Ptr;      
      typedef typename Frame<T_FeatureType, T_FeatureDescriptorType>::Ptr BaseFramePtr;
      
    public:
      MultiFrames(int seqID, int nFrames);
      
      void insertNewFrame(const BaseFramePtr& frame);
      BaseFramePtr getFrame(int index);
      
      void setMultiFramePose(Transformation multiFramePose);
      void setMultiFrameGTPose(Transformation multiFrameGTPose) { mMultiFrameGTPose = multiFrameGTPose; }
      Transformation getMultiFramePose();
      Transformation getMultiFrameGTPose() { return mMultiFrameGTPose; }
      
      int getMultiFrameID();
      
    private:
      int mnFrames;
      int mMultiFrameID;
      Transformation mMultiFramePose;
      Transformation mMultiFrameGTPose;
      std::vector<BaseFramePtr> mvFrames;
    };
  }
}
#endif