#ifndef _VGUGV_COMMON_FRAMEHPP_
#define _VGUGV_COMMON_FRAMEHPP_

#include "frame.h"
#include "device/cpu/frame_cpu.h"
#include "device/cuda/frame_cuda.h"
#include "multiFrames.h"
#include "../commDefs.h"
#include "../feature/feature_depthMap.h"

namespace VGUGV
{
  namespace Common
  {
    typedef DepthHypothesis_GMM							    DepthHypothesisBase;
    typedef Frame<Feature_depthMap<DepthHypothesisBase>, DepthHypothesisBase>       Frame_denseDepthMap;
    typedef Frame_CPU<Feature_depthMap<DepthHypothesisBase>, DepthHypothesisBase>   FrameCPU_denseDepthMap;    
    typedef MultiFrames<Feature_depthMap<DepthHypothesisBase>, DepthHypothesisBase> MultiFrames_denseDepthMap;
#ifndef COMPILE_WITHOUT_CUDA
    typedef Frame_CUDA<Feature_depthMap<DepthHypothesisBase>, DepthHypothesisBase>  FrameCUDA_denseDepthMap;
#endif
  }
}
#endif