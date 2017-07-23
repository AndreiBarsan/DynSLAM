#ifndef _VGUGV_SLAM_IMG_ALIGNMENT_DIRECT_BASE_
#define _VGUGV_SLAM_IMG_ALIGNMENT_DIRECT_BASE_

#include "../commDefs.h"
#include "../frame/frame.hpp"
#include "../cameraBase.h"
#include "../transformation/transformation.h"
#include "../robustLoss/robustLossBase.h"
#include <memory>
#include <Eigen/src/Core/util/Memory.h>

using namespace VGUGV;

namespace VGUGV
{
  namespace SLAM
  {
    class DirImgAlignBase
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
      typedef std::shared_ptr<DirImgAlignBase> Ptr;
      typedef Common::Frame_denseDepthMap           T_Frame;
      typedef Common::Frame_denseDepthMap::Ptr      T_FramePtr;
      
    public:
      DirImgAlignBase(int nMaxPyramidLevels, int nMaxIterations, float eps, Common::ROBUST_LOSS_TYPE lossType, float param);
      ~DirImgAlignBase();
      
    public:
      virtual void doAlignment(const T_FramePtr& refFrame, const T_FramePtr& curFrame, Common::Transformation& inout_Tref2cur) = 0;
           
    protected:
      int mnMaxPyramidLevels;
      int mnMaxIterations;
      float mConvergenceEps;
      Common::RobustLossBase::Ptr mRobustLossFunction;
    };
  }
}

#endif