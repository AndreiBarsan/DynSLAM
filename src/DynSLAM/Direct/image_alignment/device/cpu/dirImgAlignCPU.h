#ifndef _VGUGV_SLAM_IMG_ALIGNMENT_DIRECT_CPU_
#define _VGUGV_SLAM_IMG_ALIGNMENT_DIRECT_CPU_

// TODO(andrei): Do we need this? Ask Peidong.
//#include <Eigen/src/Core/util/Memory.h>
#include "../../dirImgAlignBase.h"
#include "../../../commDefs.h"
#include "../../../transformation/transformation.h"

namespace VGUGV
{
  namespace SLAM
  {
    class DirImgAlignCPU : public DirImgAlignBase
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
      typedef std::shared_ptr<DirImgAlignCPU> Ptr;
      using DirImgAlignBase::T_Frame;
      using DirImgAlignBase::T_FramePtr;
      
    public:
      DirImgAlignCPU(int nMaxPyramidLevels, int nMaxIterations, float eps,
					 Common::ROBUST_LOSS_TYPE type, float param, float minGradMagnitude);
      ~DirImgAlignCPU();
      
    public:
      void doAlignment(const T_FramePtr& refFrame, const T_FramePtr& curFrame, Common::Transformation& inout_Tref2cur);
      
    protected:
      void preComputeJacobianHessian(const T_FramePtr& refFrame, int level);
      
      void solverGaussNewton(const T_FramePtr& refFrame, 
			     const T_FramePtr& curFrame, 
			     int level, 
			     Eigen::Matrix4f& inout_Tref2cur);
      
      float gaussNewtonUpdateStep(const T_FramePtr& refFrame, 
				  const T_FramePtr& curFrame, 
				  int level, 
				  Eigen::Matrix4f inTref2cur, 
				  Eigen::Matrix<float, 6, 1>& outEpsilon);
      
//      float gaussNewtonUpdateStepSSE(const T_FramePtr& refFrame,
//				     const T_FramePtr& curFrame,
//				     int level,
//				     Eigen::Matrix4f inTref2cur,
//				     Eigen::Matrix<float, 6, 1>& outEpsilon);
      
    private:
      /// \brief Used for thresholding pixels based on their gradient magnitude when performing the
      ///        alignment.
      float mMinGradMagnitude;

      float* mPreCompJacobian;
      float* mPreCompHessian;
      float mAffineBrightness_a;
      float mAffineBrightness_b;
    };
  }
}


#endif