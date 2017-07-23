#ifndef _VGUGV_COMMON_ROBUST_LOSS_T_DISTRIBUTION_
#define _VGUGV_COMMON_ROBUST_LOSS_T_DISTRIBUTION_

#include "robustLossBase.h"
#include <memory>

using namespace VGUGV::Common;

namespace VGUGV
{
	namespace Common
	{
		class TDistributionLoss : public RobustLossBase
		{
		public:
			typedef std::shared_ptr<TDistributionLoss> Ptr;

		public:
			TDistributionLoss(double dof = 5.0)
				: m_Kdof(dof){}
			void setParameter(int parameter);

		public:
			float getWeight(float x);
			__m128 getWeight(const __m128& x);

		private:
			double m_Kdof;
		};
	}
}

#endif