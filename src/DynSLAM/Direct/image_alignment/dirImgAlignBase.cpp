#include "dirImgAlignBase.h"
#include "../robustLoss/pseudoHuberLoss.h"
#include "../robustLoss/tDistributionLoss.h"

namespace VGUGV
{
  namespace SLAM
  {
    DirImgAlignBase::DirImgAlignBase(int nMaxPyramidLevels, int nMaxIterations, float eps, Common::ROBUST_LOSS_TYPE lossType, float param)
    : mnMaxPyramidLevels(nMaxPyramidLevels)
    , mnMaxIterations(nMaxIterations)
    , mConvergenceEps(eps)
    {
      switch(lossType)
      {
	case Common::ROBUST_LOSS_TYPE::PSEUDO_HUBER:
	  mRobustLossFunction = std::make_shared<Common::PseudoHuberLoss>(param);
	  break;
	case Common::ROBUST_LOSS_TYPE::TDISTRIBUTION:
	  mRobustLossFunction = std::make_shared<Common::TDistributionLoss>(param);
	  break;
	default:
	  break;
      }
    }
    
    DirImgAlignBase::~DirImgAlignBase()
    {}    
  }
}
