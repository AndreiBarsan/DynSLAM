#ifndef _VGUGV_COMMON_ROBUST_LOSS_PSEUDO_HUBER_
#define _VGUGV_COMMON_ROBUST_LOSS_PSEUDO_HUBER_

#include "robustLossBase.h"
#include <memory>

using namespace VGUGV::Common;

namespace VGUGV
{
	namespace Common
	{
		class PseudoHuberLoss : public RobustLossBase
		{
		public:
			typedef std::shared_ptr<PseudoHuberLoss> Ptr;

		public:
			PseudoHuberLoss(double delta)
				: mDelta2Inv(1.0f / (delta*delta)){}

			void setParameter(int parameter);

		public:
			float getWeight(float x);
			__m128 getWeight(const __m128& x);

		private:
			float mDelta2Inv;
		};
	}
}

#endif