#ifndef _VGUGV_COMMON_HELPER_FUNCTIONS__
#define _VGUGV_COMMON_HELPER_FUNCTIONS_

#include <chrono>
#include <cmath>
#include "commDefs.h"
#include <Eigen/Dense>
#include "cameraBase.h"
#include "cudaDefs.h"

namespace VGUGV
{
  namespace Common
  {
    typedef long long T_int64;
    typedef std::chrono::high_resolution_clock T_clock;
    typedef std::chrono::time_point<T_clock> T_time;
    
    /*-------------------------------------------------------------------------------------------------*/
    /*------------------------------------ FUNCTION PROTOTYPES ----------------------------------------*/
    /*-------------------------------------------------------------------------------------------------*/
    
    template<typename T> T INPI(T angle);
    template<typename T> T deg2rad(T deg);
    template<typename T> T rad2deg(T rad);
    template<typename T> void rotMatrix2Euler(T r00, T r01, T r02, 
					   T r10, T r11, T r12,
					   T r20, T r21, T r22,
					   T& phi, T& theta, T& psi);
    template<typename T> void euler2rotMatrix(T phi, T theta, T psi,
					      T& r00, T& r01, T& r02, 
					      T& r10, T& r11, T& r12,
					      T& r20, T& r21, T& r22);
    
    T_time   currentTime();
    T_int64  elapsedTime(T_time start, TIME_TYPE type);
    
    Eigen::Matrix3f so3Exp(Eigen::Vector3f omega);
    Eigen::Matrix3f fov2K(float fovDeg, int pixelWidth, int pixelHeight);
    Eigen::Vector2f distort_radialNtangential(CameraBase::Ptr cameraModel, Eigen::Vector2f pixel0);
    Eigen::Matrix3f computePlanarHomography(Eigen::Matrix4f T_l2r, float planeDepth_ref, Eigen::Vector3f planeNormal_ref);
    
    float  bilinearInterpolation(const unsigned char* pData, int nRows, int nCols, float r, float c); 
    __m128 bilinearInterpolation(const unsigned char* pData, int nRows, int nCols, const SSE_m128_v2& pixels);
    bool getImagePatch(unsigned char* pImageData, int nRows, int nCols, int r, int c, int nPatchSize, unsigned char* pPatch);
    bool getImagePatch(CameraBase::Ptr pRefCamera, 
		       CameraBase::Ptr pCurCamera, 
		       unsigned char* pCurImgData, 
		       Eigen::Matrix3f Hl2r, 
		       int r, 
		       int c, 
		       int nPatchSize, 
		       unsigned char* pPatch);
    
    float znccScore(unsigned char* pRefPatch, unsigned char* pCurPatch, int nSize);
    
    //! Method to cluster 1D data into different clusters, each cluster is approximated by a Gaussian distribution
      /*!
      \param inputData (data, weight), data is already sorted ascendingly. 
      \param dataSize total number of data pairs 
      \param Th threshold value used to split two clusters
      \param models fitted Gaussian models (mean, variance, weight) 
      \return (nModels, nBestModelIndex)
      */
    Eigen::Vector2i cluster_1Ddata(Eigen::Vector2f* inputData, int dataSize, float Th, Eigen::Vector3f* models);
    
    //! Method to fit 1D data into a Gaussian distribution
      /*!
      \param inputData (data, weight), data is already sorted ascendingly. 
      \param dataSize total number of data pairs  
      \return (mean, variance, weight)
      */
    Eigen::Vector3f gaussian_fit(Eigen::Vector2f* inputData, int dataSize);
    
    float depthFromSubpixelInterpolation(const std::array<float, 3>& depths, const std::array<float, 3>& similarityScores);
    
    int intergerDivUp(int a, int b);
    
    // SSE related functions
    void printSSE_m128(const __m128& a);
    __m128 _mm_floor_ps2(const __m128& a);
  }
}
#endif