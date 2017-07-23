#include "tDistributionLoss.h"

#include <cmath>
#include <iostream>

using namespace VGUGV::Common;

float TDistributionLoss::getWeight(float x)
{
  return (m_Kdof + 1.0f) / (m_Kdof + x * x);
}

__m128 TDistributionLoss::getWeight(const __m128& x)
{
  __m128 x2 = _mm_mul_ps(x, x);
  __m128 sse_kDof = _mm_set1_ps(m_Kdof);
  __m128 sse_Ones = _mm_set1_ps(1.0);
  return _mm_div_ps(_mm_add_ps(sse_kDof, sse_Ones), _mm_add_ps(sse_kDof, x2));
}

void TDistributionLoss::setParameter(int parameter)
{
	m_Kdof = parameter;
}