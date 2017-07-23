#include "pinholeCameraModel.h"
#include <iostream>

namespace VGUGV
{
  namespace Common
  {
    PinholeCameraModel::~PinholeCameraModel()
    {
      delete [] mRays; mRays = NULL;
      #ifndef DCOMPILE_WITHOUT_CUDA
      if(mRays_CUDA != NULL) CUDA_SAFE_CALL(cudaFree(mRays_CUDA));
      #endif
    }
    
    SSE_m128_v2 PinholeCameraModel::project(SSE_m128_v3 scenePoints)
    {
      __m128 sse_normPoints_x = scenePoints.m[0] / scenePoints.m[2];
      __m128 sse_normPoints_y = scenePoints.m[1] / scenePoints.m[2];
      
      __m128 px = _mm_add_ps(_mm_set1_ps(mK(0, 2)), _mm_mul_ps(_mm_set1_ps(mK(0, 0)), sse_normPoints_x));
      __m128 py = _mm_add_ps(_mm_set1_ps(mK(1, 2)), _mm_mul_ps(_mm_set1_ps(mK(1, 1)), sse_normPoints_y));
      
      SSE_m128_v2 temp;
      temp.m[0] = px;
      temp.m[1] = py;
      return temp;
    }
		  
    bool PinholeCameraModel::project(const Eigen::Vector3f& scenePoint, Eigen::Vector2f& pixelPoint, float* jacobians)
    {
      if (scenePoint(2) < 1e-3)
      {
	return false;
      }
      
      // project scene point to normalized image plane
      Eigen::Vector3f normalizedPoint = scenePoint / scenePoint(2);
      pixelPoint = (mK*normalizedPoint).block<2, 1>(0, 0);
      
      // check the pixel lies in bound
      int nRows = mImageSize(0);
      int nCols = mImageSize(1);
      
      if (pixelPoint(0) < 0 || pixelPoint(0) > nCols - 1 || pixelPoint(1) < 0 || pixelPoint(1) > nRows - 1)
      {
	return false;
      }
      return true;
    }
    
    bool PinholeCameraModel::backProject(const Eigen::Vector2f& pixelPoint, Eigen::Vector3f& ray)
    {
      // check the pixel lies in bound
      int nRows = mImageSize(0);
      int nCols = mImageSize(1);
      if (pixelPoint(0) < 0 || pixelPoint(0) > nCols - 1 || pixelPoint(1) < 0 || pixelPoint(1) > nRows - 1)
      {
	return false;
      }
      
      float u = mKinv(0, 0)*pixelPoint(0) + mKinv(0, 2);
      float v = mKinv(1, 1)*pixelPoint(1) + mKinv(1, 2);
      ray << u, v, 1.0f;
      ray.normalize();
      
      return true;
    }
    
    
    bool PinholeCameraModel::backProject(int row, int col, Eigen::Vector3f& ray)
    {
      if (mRays == NULL)
      {
	mRays = new Common::Vector3<float>[mImageSize[0] * mImageSize[1]];
	for (int r = 0; r < mImageSize(0); r++)
	{
	  for (int c = 0; c < mImageSize(1); c++)
	  {
	    int index = r*mImageSize(1) + c;
	    Eigen::Vector2f pixel; pixel << c, r;
	    Eigen::Vector3f rayTmp;
	    backProject(pixel, rayTmp);
	    
	    Common::Vector3<float> temp(rayTmp(0), rayTmp(1), rayTmp(2));
	    mRays[index] = temp;
	  }
	}
      }
      int index = row*mImageSize(1) + col;
      ray(0) = mRays[index].x;
      ray(1) = mRays[index].y;
      ray(2) = mRays[index].z;
      return true;
    }
    
    bool PinholeCameraModel::projectionJacobian(const Eigen::Vector3f& scenePoint, int pyramidLevel, Eigen::Matrix<float, 2, 3>& jacobian)
    {
      float x = scenePoint(0); float y = scenePoint(1); float z = scenePoint(2);
      if (fabs(z) < 1e-3) return false;
      // projection from 3D landmark to normalized image plane
      float iz = 1.0f/z;
      float iz2 = iz*iz;
      
      Eigen::Matrix<float, 2, 3> J1;
      J1(0, 0) = iz;   J1(0, 1) = 0.0f; J1(0, 2) = (-1.0f)*x*iz2;
      J1(1, 0) = 0.0f; J1(1, 1) = iz; J1(1, 2) = (-1.0f)*y*iz2;
      
      // from normalized image plane to pixel
      jacobian = J1;
      jacobian.row(0) = jacobian.row(0)*mK(0, 0);
      jacobian.row(1) = jacobian.row(1)*mK(1, 1);
      
      // from pyramid level0 to pyramid pyramidLevel
      float scale = 1.0f / (1 << pyramidLevel);
      jacobian = jacobian * scale;
      
      return true;
    }
    
    // retrieve
    Eigen::Vector2i PinholeCameraModel::getCameraSize()
    {
      return mImageSize;
    }
    
    void PinholeCameraModel::getK(Eigen::Matrix3f& K)
    {
      K = mK;
    }
    
    void PinholeCameraModel::getK(float& fx, float& fy, float& cx, float& cy)
    {
      fx = mK(0, 0);
      fy = mK(1, 1);
      cx = mK(0, 2);
      cy = mK(1, 2);
    }
    
    void PinholeCameraModel::getKinv(Eigen::Matrix3f& Kinv)
    {
      Kinv = mKinv;
    }
    
    void PinholeCameraModel::getKinv(float& fxInv, float& fyInv, float& cxInv, float& cyInv)
    {
      fxInv = mKinv(0, 0);
      fyInv = mKinv(1, 1);
      cxInv = mKinv(0, 2);
      cyInv = mKinv(1, 2);
    }
    
    void PinholeCameraModel::getDistoritionParams(std::vector<float>&)
    {
      assert(1 == 0);
      std::cout << "Woops, it seems I don't have the implementation for this function...";
    }
    
    void PinholeCameraModel::getAdditionalParams(std::vector<float>& params)
    {
      params.clear();
    }
    
    Vector3<float>* PinholeCameraModel::getRayPtrs(Common::DEVICE_TYPE device)
    {
      // make sure look-up table has been created...
      Eigen::Vector3f temp;
      backProject(0, 0, temp);
      //
      if(device == Common::DEVICE_TYPE::CPU)
      {
	return mRays;
      }
      else
      {
#ifndef DCOMPILE_WITHOUT_CUDA
	if(mRays_CUDA == NULL)
	{
	  CUDA_SAFE_CALL(cudaMalloc(&mRays_CUDA, sizeof(Vector3<float>) * mImageSize(0) * mImageSize(1)));
	  CUDA_SAFE_CALL(cudaMemcpy(mRays_CUDA, mRays, sizeof(Vector3<float>) * mImageSize(0) * mImageSize(1), cudaMemcpyHostToDevice));
	}
	return mRays_CUDA;
#endif
      }
    }
    
    // set 
    void PinholeCameraModel::setDistortionParams(std::vector<float>)
    {
      assert(1 == 0);
      std::cout << "Woops, it seems I don't have the implementation for this function...";
    }
  }
}