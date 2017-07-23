#include "pseudoHuberLoss.h"

using namespace VGUGV::Common;

float PseudoHuberLoss::getWeight(float x)
{
  return 1.0f / sqrt(1.0f + x*x*mDelta2Inv);
}

__m128 PseudoHuberLoss::getWeight(const __m128& x)
{
  __m128 denominator = _mm_sqrt_ps(_mm_add_ps(_mm_set1_ps(1.0), _mm_mul_ps(_mm_mul_ps(x,x), _mm_set1_ps(mDelta2Inv))));
  return _mm_div_ps(_mm_set1_ps(1.0), denominator);
}

void PseudoHuberLoss::setParameter(int parameter)
{
	mDelta2Inv = 1.0f / (parameter*parameter);
}