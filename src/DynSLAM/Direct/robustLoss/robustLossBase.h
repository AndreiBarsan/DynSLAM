#ifndef _VGUGV_COMMON_ROBUST_LOSS_
#define _VGUGV_COMMON_ROBUST_LOSS_

#include <memory>
#include "../commDefs.h"

namespace VGUGV
{
  namespace Common
  {
    class RobustLossBase
    {
    public:
      typedef std::shared_ptr<RobustLossBase> Ptr;      
    public:
      virtual float getWeight(float x) = 0;
      virtual __m128 getWeight(const __m128& x) = 0;
      virtual void setParameter(int parameter) = 0;
    };
  }
}

#endif