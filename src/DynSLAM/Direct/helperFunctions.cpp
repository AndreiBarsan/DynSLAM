#include "helperFunctions.hpp"
#include <smmintrin.h>
#include <iostream>

namespace VGUGV
{
  namespace Common
  {
    template<typename T> T INPI(T angle) { return (angle - floor((angle + M_PI) / (2 * M_PI)) * 2 * M_PI);}
    template<typename T> T deg2rad(T deg) { return (deg*M_PI) / 180.0f;}
    template<typename T> T rad2deg(T rad) { return (rad*180.0f) / M_PI;}
    template<typename T> void rotMatrix2Euler(T r00, T r01, T r02, 
					   T r10, T r11, T r12,
					   T r20, T r21, T r22,
					   T& phi, T& theta, T& psi)
    {
      float sy = sqrt(r00 * r00 +  r10 * r10);
      bool singular = sy < 1e-6; 
      
      if (!singular)
      {
	phi = atan2(r21 , r22);
	theta = atan2(-r20, sy);
	psi = atan2(r10, r00);
      }
      else
      {
	phi = atan2(-r12, r11);
	theta = atan2(-r20, sy);
	psi = 0;
      }
    }
    
    template<typename T> void euler2rotMatrix(T phi, T theta, T psi,
					      T& r00, T& r01, T& r02, 
					      T& r10, T& r11, T& r12,
					      T& r20, T& r21, T& r22)
    {
      // Calculate rotation about x axis
      Eigen::Matrix<T, 3, 3> R_x;
      R_x <<   1,       0,              0,
               0,       cos(phi),   -sin(phi),
               0,       sin(phi),   cos(phi);
	       
      // Calculate rotation about y axis
      Eigen::Matrix<T, 3, 3> R_y;
      R_y << cos(theta),    0,      sin(theta),
               0,           1,      0,
               -sin(theta), 0,      cos(theta);
	       
      // Calculate rotation about z axis
      Eigen::Matrix<T, 3, 3> R_z;
      R_z << cos(psi),    -sin(psi),      0,
             sin(psi),    cos(psi),       0,
             0,               0,          1;
     // Combined rotation matrix
     Eigen::Matrix<T, 3, 3> R = R_z * R_y * R_x;
     r00 = R(0,0); r01 = R(0,1); r02 = R(0,2);
     r10 = R(1,0); r11 = R(1,1); r12 = R(1,2);
     r20 = R(2,0); r21 = R(2,1); r22 = R(2,2);
    }
    
    template float rad2deg<float>(float);
    template float deg2rad<float>(float);
    template float INPI<float>(float);
    template void  rotMatrix2Euler<float>(float, float, float,
					  float, float, float,
					  float, float, float,
					  float&, float&, float&);
    template void  euler2rotMatrix<float>(float, float, float,
					  float&, float&, float&,
					  float&, float&, float&,
					  float&, float&, float&);
    
    T_time currentTime() { return T_clock::now();}
    T_int64 elapsedTime(T_time start, TIME_TYPE type)
    {
      auto end = T_clock::now();
      auto diff = (end - start);
      switch(type)
      {
	case TIME_TYPE::US:
	  return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
	case TIME_TYPE::MS:
	  return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
	case TIME_TYPE::S:
	  return std::chrono::duration_cast<std::chrono::seconds>(diff).count();
      }
    }
    
    Eigen::Matrix3f so3Exp(Eigen::Vector3f omega)
    {
      Eigen::Matrix3f SO3;
      float wx = omega(0); float wy = omega(1); float wz = omega(2);
      float theta = sqrt(wx*wx + wy*wy + wz*wz);
      Eigen::Matrix3f Omega;
      Omega(0, 0) = 0.0f;  Omega(0, 1) = -wz;  Omega(0, 2) = wy;
      Omega(1, 0) = wz;    Omega(1, 1) = 0.0f; Omega(1, 2) = -wx;
      Omega(2, 0) = -wy;   Omega(2, 1) = wx;   Omega(2, 2) = 0.0f;
      
      if (theta < 1e-10)
      {
	SO3 = Eigen::Matrix3f::Identity();
      }
      else
      {
	SO3 = Eigen::Matrix3f::Identity() + Omega*sin(theta) / theta + Omega*Omega*(1 - cos(theta)) / (theta*theta);
      }
      return SO3;
    }
    
    Eigen::Matrix3f fov2K(float fovDeg, int pixelWidth, int pixelHeight)
    {
      float aspectRatio = static_cast<float>(pixelHeight) / static_cast<float>(pixelWidth);
      float widthInMeter = 2.0f*tan(deg2rad(fovDeg) / 2.0f);
      float heightInMeter = widthInMeter*aspectRatio;
      
      float fx = static_cast<float>(pixelWidth) / widthInMeter;
      float fy = static_cast<float>(pixelHeight) / heightInMeter;
      
      float cx = static_cast<float>(pixelWidth) / 2.0f;
      float cy = static_cast<float>(pixelHeight) / 2.0f;
      
      Eigen::Matrix3f K;
      K << fx, 0.0f, cx,
      0.0f, fy, cy,
      0.0f, 0.0f, 1.0f;
      return K;
    }
    
    Eigen::Vector2f distort_radialNtangential(CameraBase::Ptr cameraModel, Eigen::Vector2f pixel0)
    {
      // distCoeff follows OpenCV order, i.e., k1, k2, p1, p2, k3
      
      // normalize pixel position by K^(-1)
      Eigen::Matrix3f K;
      Eigen::Matrix3f Kinv;
      
      cameraModel->getK(K);
      cameraModel->getKinv(Kinv);
      
      Eigen::Vector3f pixel0H = Eigen::Vector3f::Ones();
      pixel0H.block<2, 1>(0, 0) = pixel0;
      
      Eigen::Vector2f normalizedPixel0 = (Kinv*pixel0H).block<2, 1>(0, 0);
      
      // add radial distortion
      float mu = normalizedPixel0(0);
      float mv = normalizedPixel0(1);
      
      // add radial distortion
      std::vector<float> distCoeff;
      cameraModel->getDistoritionParams(distCoeff);
      if (distCoeff.size() == 0)
      {
	distCoeff.resize(5, 0);
      }
      
      float k1 = distCoeff.at(0);
      float k2 = distCoeff.at(1);
      float p1 = distCoeff.at(2);
      float p2 = distCoeff.at(3);
      float k3 = distCoeff.at(4);
      
      float rho = sqrt(mu*mu + mv*mv);
      float rho2 = rho*rho;
      float rho4 = rho2*rho2;
      
      float distortion = 1 + k1*rho2 + k2*rho4 + k3*rho2*rho4;
      
      float mdu = mu*distortion;
      float mdv = mv*distortion;
      
      // add tangential distortion
      mdu = mdu + 2 * p1*mdu*mdv + p2*(rho2 + 2 * mdu*mdu);
      mdv = mdv + p1*(rho2 + 2 * mdv*mdv) + 2 * p2*mdu*mdv;
      
      //
      Eigen::Vector2f distortedPixel;
      distortedPixel(0) = K(0, 0)*mdu + K(0, 2);
      distortedPixel(1) = K(1, 1)*mdv + K(1, 2);
      return distortedPixel;
    }

    Eigen::Matrix3f computePlanarHomography(Eigen::Matrix4f T_l2r, float planeDepth_ref, Eigen::Vector3f planeNormal_ref)
    {
      return T_l2r.block<3, 3>(0, 0) - (T_l2r.block<3, 1>(0, 3) * planeNormal_ref.transpose()) / planeDepth_ref;
    }
    
    __m128 bilinearInterpolation(const unsigned char* pData, int nRows, int nCols, const SSE_m128_v2& pixels)
    {
      __m128 sse_x  = _mm_floor_ps2(pixels.m[0]);
      __m128 sse_y  = _mm_floor_ps2(pixels.m[1]);
      __m128 sse_dx = _mm_sub_ps(pixels.m[0], sse_x);
      __m128 sse_dy = _mm_sub_ps(pixels.m[1], sse_y);
      __m128 sse_dxdy = _mm_mul_ps(sse_dx, sse_dy);
      
      __m128 sse_w00 = _mm_add_ps(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1.0), sse_dx), sse_dy), sse_dxdy);
      __m128 sse_w01 = _mm_sub_ps(sse_dx, sse_dxdy);
      __m128 sse_w10 = _mm_sub_ps(sse_dy, sse_dxdy);
      __m128 sse_w11 = sse_dxdy;
      
      const __m128 sse_nCols = _mm_set1_ps(nCols);
      
      __m128 sse_index00 = _mm_add_ps(_mm_mul_ps(sse_y, sse_nCols), sse_x);      
      // load pixel intensity
      float index00f[4]; _mm_store_ps(index00f, sse_index00);
      int   index00[4];
      index00[0] = index00f[0] + 0.5f;
      index00[1] = index00f[1] + 0.5f;
      index00[2] = index00f[2] + 0.5f;
      index00[3] = index00f[3] + 0.5f;
      
      float intensity00[4];
      intensity00[0] = pData[index00[0]];
      intensity00[1] = pData[index00[1]];
      intensity00[2] = pData[index00[2]];      
      intensity00[3] = pData[index00[3]];
      
      float intensity01[4];
      intensity01[0] = pData[index00[0] + 1];
      intensity01[1] = pData[index00[1] + 1];
      intensity01[2] = pData[index00[2] + 1];      
      intensity01[3] = pData[index00[3] + 1];

      float intensity10[4];
      intensity10[0] = pData[index00[0] + nCols];
      intensity10[1] = pData[index00[1] + nCols];
      intensity10[2] = pData[index00[2] + nCols];      
      intensity10[3] = pData[index00[3] + nCols];

      float intensity11[4];
      intensity11[0] = pData[index00[0] + nCols + 1];
      intensity11[1] = pData[index00[1] + nCols + 1];
      intensity11[2] = pData[index00[2] + nCols + 1];      
      intensity11[3] = pData[index00[3] + nCols + 1];

      // compute interpolated intensity
      __m128 sse_intensity00 = _mm_load_ps(intensity00);
      __m128 sse_intensity01 = _mm_load_ps(intensity01);
      __m128 sse_intensity10 = _mm_load_ps(intensity10);
      __m128 sse_intensity11 = _mm_load_ps(intensity11);
      
      // 
      return _mm_add_ps(_mm_add_ps(_mm_mul_ps(sse_w00, sse_intensity00), _mm_mul_ps(sse_w01, sse_intensity01)), 
			_mm_add_ps(_mm_mul_ps(sse_w10, sse_intensity10), _mm_mul_ps(sse_w11, sse_intensity11)));			
    }
	
    float bilinearInterpolation(const unsigned char* pData, int nRows, int nCols, float row, float col)
    {
      int rx = static_cast<int>(col + 0.5f);
      int ry = static_cast<int>(row + 0.5f);

      if (rx < 0 || rx >= nCols || ry < 0 || ry >= nRows) {
//        throw std::runtime_error("Cannot bilinearly interpolate outside the image.");
        std::cerr <<"Cannot bilinearly interpolate outside the image." << std::endl;
      }
      
      if (ry == 0 || ry == nRows - 1 || rx == 0 || rx == nCols - 1)
      {
        return pData[ry*nCols + rx];
      }
      
      int y = static_cast<int>(row);
      int x = static_cast<int>(col);
      
      float dy = row - y;
      float dx = col - x;
      float dxdy = dx*dy;
      
      float w00 = 1.0f - dx - dy + dxdy;
      float w01 = dx - dxdy;
      float w10 = dy - dxdy;
      float w11 = dxdy;
      
      const int baseIndex = y * nCols + x;
      const unsigned char p00 = pData[baseIndex];
      const unsigned char p01 = pData[baseIndex + 1];
      const unsigned char p10 = pData[baseIndex + nCols];
      const unsigned char p11 = pData[baseIndex + nCols + 1];

//      std::cout << "Bilinear interpolation at row = " << row << ", col = " << col << " for "
//                << nRows << " total rows and " << nCols << " total cols. p's: "
//                << (int)p00 << ", " << (int)p01 << ", " << (int)p10 << ", " << (int)p11 << " with weights "
//                << w00 << ", " << w01 << ", " << w10 << ", " << w11 << "." << std::endl;
      
      float value = w11 * p11 + w10 * p10 + w01 * p01 + w00 * p00;
      return value;
    }
    
    bool getImagePatch(unsigned char* pImageData, int nRows, int nCols, int r, int c, int nPatchSize, unsigned char* pPatch)
    {
      int halfSize = nPatchSize * 0.5f;
      int min_r = r - halfSize;
      int min_c = c - halfSize;
      int max_r = r + halfSize + 1;
      int max_c = c + halfSize + 1;
      if(min_r < 0 || min_c < 0 || max_r > nRows || max_c > nCols) return false;
      
      for(int i = 0; i < nPatchSize; i++)
      {
	unsigned char* pImageSrc = pImageData + (min_r + i) * nCols + min_c;
	memcpy(pPatch + nPatchSize * i, pImageSrc, nPatchSize);
      }
    }
    
    bool getImagePatch(CameraBase::Ptr pRefCamera, 
		       CameraBase::Ptr pCurCamera, 
		       unsigned char* pCurImgData, 
		       Eigen::Matrix3f Hl2r, 
		       int r, 
		       int c, 
		       int nPatchSize, 
		       unsigned char* pPatch)
    {
      
      int halfSize = nPatchSize * 0.5f;
      int min_r = r - halfSize;
      int min_c = c - halfSize;
      int max_r = r + halfSize + 1;
      int max_c = c + halfSize + 1;
      
      Eigen::Vector2i imageSize = pRefCamera->getCameraSize();
      int nRows = imageSize(0); int nCols = imageSize(1);
      if(min_r < 0 || min_c < 0 || max_r > nRows || max_c > nCols) return false;
      
      int index = 0;
      for(int row = min_r; row < max_r; row++)
      {
	for(int col = min_c; col < max_c; col++, index++)
	{
	  Eigen::Vector3f unitRefRay;
	  pRefCamera->backProject(row, col, unitRefRay);
	  
	  Eigen::Vector3f unitCurRay = Hl2r * unitRefRay;
	  Eigen::Vector2f curPixel;
	  if(!pCurCamera->project(unitCurRay, curPixel)) return false;
	  
	  float intensity = bilinearInterpolation(pCurImgData, nRows, nCols, curPixel(1), curPixel(0));
	  pPatch[index] = intensity;
	}
      }
      return true;
    }
    
    float znccScore(unsigned char* pRefPatch, unsigned char* pCurPatch, int nPixels)
    {
      unsigned int sum1 = 0, sumSquares1 = 0, sum2 = 0, sumSquares2 = 0;
      for (int i = 0; i < nPixels; i++)
      {
	unsigned int p1 = static_cast<unsigned int>(pRefPatch[i]);
	unsigned int p2 = static_cast<unsigned int>(pCurPatch[i]);
	
	sum1 += p1; sumSquares1 += p1*p1;
	sum2 += p2; sumSquares2 += p2*p2;
      }
      
      float u1 = static_cast<float>(sum1) / static_cast<float>(nPixels);
      float u2 = static_cast<float>(sum2) / static_cast<float>(nPixels);
      
      float numerator = 0.0f;
      for (int i = 0; i < nPixels; i++)
      {
	float d1 = static_cast<float>(pRefPatch[i]) - u1;
	float d2 = static_cast<float>(pCurPatch[i]) - u2;
	numerator += (d1*d2);
      }
      
      float var1 = static_cast<float>(sumSquares1)-static_cast<float>(sum1*sum1) / static_cast<float>(nPixels);
      float var2 = static_cast<float>(sumSquares2)-static_cast<float>(sum2*sum2) / static_cast<float>(nPixels);
      
      // assume affine model 
      float ratio = u2 / u1;
      float ratio2 = ratio * ratio;
      
      if (fabs(var2 / (ratio2 * var1) - 1.0f) > 0.9) return -1.0f;
      return  numerator / sqrt(var1*var2);
    }
    
    Eigen::Vector2i cluster_1Ddata(Eigen::Vector2f* inputData, int nDataSize, float Th, Eigen::Vector3f* models)
    {
      // mean, variance
      int   bestModelIndex = 0;
      float bestWeight = 0.0f;
      
      std::vector<Eigen::Vector2f> cluster; // data, weight
      std::vector<Eigen::Vector3f> vector_models;
      for(int i = 0; i < nDataSize; i++)
      {
	Eigen::Vector2f curData  = inputData[i];
	
	if(i == 0)
	{
	  if(i != nDataSize - 1) cluster.push_back(curData);
	  if(nDataSize == 1 && cluster.size() > 0)
	  {
	    Eigen::Vector3f model = gaussian_fit(cluster.data(), cluster.size());
	    bestWeight = model(2);
	    bestModelIndex = vector_models.size();
	    vector_models.push_back(model);
	  }
	  continue;
	}
	
	Eigen::Vector2f prevData = inputData[i-1];
	
	if(fabs(curData(0) - prevData(0)) < Th)
	{
	  if(i != nDataSize - 1) cluster.push_back(curData);
	}
	else if(cluster.size() > 0)
	{
	  Eigen::Vector3f model = gaussian_fit(cluster.data(), cluster.size());
	  if (model(2) > bestWeight)
	  {
	    bestWeight = model(2);
	    bestModelIndex = vector_models.size();
	  }
	  vector_models.push_back(model);
	  
	  cluster.clear();
	  if(i != nDataSize - 1) cluster.push_back(curData);
	}
	
	if(i == nDataSize - 1 && cluster.size() > 0)
	{
	  Eigen::Vector3f model = gaussian_fit(cluster.data(), cluster.size());
	  if (model(2) > bestWeight)
	  {
	    bestWeight = model(2);
	    bestModelIndex = vector_models.size();
	  }
	  vector_models.push_back(model);
	}
      }
      memcpy(models, vector_models.data(), sizeof(Eigen::Vector3f) * vector_models.size());
      return Eigen::Vector2i(vector_models.size(), bestModelIndex);
    }
    
    Eigen::Vector3f gaussian_fit(Eigen::Vector2f* inputData, int nDataSize)
    {
      float sumOfWeight = 0.0f;
      float sumOfWeightedData = 0.0f;
      float bestWeight = 0.0f;
      Eigen::Vector3f model;
      for(int i = 0; i < nDataSize; i++)
      {
	Eigen::Vector2f data = inputData[i];
	sumOfWeight += data(1);
	sumOfWeightedData += (data(0) * data(1));
	if(data(1) > bestWeight)
	{
	  bestWeight = data(1);
	}
      }
      model(0) = sumOfWeightedData / sumOfWeight;
      model(2) = bestWeight;
      
      float sumOfMeanErrors = 0.0f;
      for(int i = 0; i < nDataSize; i++)
      {
	Eigen::Vector2f data = inputData[i];
	sumOfMeanErrors += (data(1) * (data(0) - model(0)) * (data(0) - model(0)));
      }
      model(1) = sumOfMeanErrors / sumOfWeight;
      model(1) = std::max(model(1), 0.000001f);
      return model;
    }
    
    float depthFromSubpixelInterpolation(const std::array<float, 3>& depths, const std::array<float, 3>& similarityScores)
    {
	if (similarityScores[1] < similarityScores[0] || similarityScores[1] < similarityScores[2])
	{
		if (similarityScores[0] > similarityScores[2])
		{
			return depths[0];
		}
		else
		{
			return depths[2];
		}
	}

	// Use the quadratic estimator to estimate the offset referenced to the middle sample.
	float offset = 0.0f;

	float denom = 2.0f * (2.0f * similarityScores[1] - similarityScores[0] - similarityScores[2]);
	if (denom > 1e-5)
	{
		offset = (similarityScores[2] - similarityScores[0]) / denom;
	}

	float invDepthOffset = 0.0;
	if (offset < 0.0f)
	{
		invDepthOffset = (1.0 / depths[1] - 1.0 / depths[0]) * offset;
	}
	else
	{
		invDepthOffset = (1.0 / depths[2] - 1.0 / depths[1]) * offset;
	}
	return 1.0 / (1.0 / depths[1] + invDepthOffset);
    }

    int intergerDivUp(int a, int b)
    {
      return ((a % b) != 0) ? (a / b + 1) : (a / b);
    }
    
    void printSSE_m128(const __m128& a)
    {
      float _a[4];
      _mm_store_ps(_a, a);
      printf("%f %f %f %f\n", _a[0], _a[1], _a[2], _a[3]);
    }
    
    __m128 _mm_floor_ps2(const __m128& a)
    {
      __m128 v0 = _mm_setzero_ps();
      __m128 v1 = _mm_cmpeq_ps(v0,v0);
      __m128i vA = _mm_srli_epi32( *(__m128i*)&v1, 9);
      __m128i vB = _mm_srli_epi32( *(__m128i*)&v1, 26);
      vB = _mm_slli_epi32( vB, 24);
      __m128 vNearest1 = _mm_or_ps(*(__m128*)&vA, *(__m128*)&vB);
      __m128i i = _mm_cvttps_epi32(a);
      __m128 aTrunc = _mm_cvtepi32_ps(i);         // truncate a
      __m128 rmd = _mm_sub_ps(a, aTrunc);         // get remainder
      __m128 rmd2 = _mm_sub_ps( rmd, vNearest1);  // sub remainder by near 1 will yield the needed offset
      __m128i rmd2i = _mm_cvttps_epi32(rmd2);     // after being truncated of course
      __m128 rmd2Trunc = _mm_cvtepi32_ps(rmd2i);
      __m128 r =_mm_add_ps(aTrunc, rmd2Trunc); 
      return r;
    }
  }
}

