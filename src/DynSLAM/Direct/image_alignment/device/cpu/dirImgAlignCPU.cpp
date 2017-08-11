#include "dirImgAlignCPU.h"
#include "../../../helperFunctions.hpp"

#include <sophus/se3.hpp>
#include <pmmintrin.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "../../../commDefs.h"
#include "../../../cameraBase.h"
#include "../../../frame/frame.hpp"

namespace VGUGV {
namespace SLAM {
DirImgAlignCPU::DirImgAlignCPU(int nMaxPyramidLevels,
                               int nMaxIterations,
                               float eps,
                               Common::ROBUST_LOSS_TYPE type,
                               float param,
                               float minGradMagnitude)
    : DirImgAlignBase(nMaxPyramidLevels, nMaxIterations, eps, type, param),
      mPreCompJacobian(NULL),
      mPreCompHessian(NULL),
      mMinGradMagnitude(minGradMagnitude)
{}

DirImgAlignCPU::~DirImgAlignCPU() {}

void DirImgAlignCPU::doAlignment(const T_FramePtr &refFrame,
                                 const T_FramePtr &curFrame,
                                 Common::Transformation &inout_Tref2cur) {
  mAffineBrightness_a = 1.0;
  mAffineBrightness_b = 0.0;

  refFrame->computeImagePyramids(mnMaxPyramidLevels);
  curFrame->computeImagePyramids(mnMaxPyramidLevels);

  Eigen::Matrix4f T_ref2cur = inout_Tref2cur.getTMatrix();
  for (int i = mnMaxPyramidLevels - 1; i >= 0; i--) {
    std::cout << std::endl << "Aligning @ level " << (i+1) << "." << std::endl;
    int nNumValidatedPixels = refFrame->getFeatureSize(i);
    mPreCompJacobian = new float[nNumValidatedPixels * 6];
    mPreCompHessian = new float[nNumValidatedPixels * 36];

    memset(mPreCompJacobian, 0, sizeof(float) * nNumValidatedPixels * 6);
    memset(mPreCompHessian, 0, sizeof(float) * nNumValidatedPixels * 36);

    // Pre-computing the Jacobian and Hessian before the actual optimization improves performance.
    preComputeJacobianHessian(refFrame, i);
    solverGaussNewton(refFrame, curFrame, i, T_ref2cur);

    delete[] mPreCompJacobian;
    delete[] mPreCompHessian;
    mPreCompJacobian = NULL;
    mPreCompHessian = NULL;
  }
  inout_Tref2cur.setT(T_ref2cur);
}

void DirImgAlignCPU::preComputeJacobianHessian(const T_FramePtr &refFrame, int level) {
  using namespace std;
  cout << "Precomputing jac and hess @ level " << (level + 1) << "." << endl;

  int scale = 1 << level;
//  int nRows = refFrame->getFrameSize()(0) / scale;
  int nCols = refFrame->getFrameSize()(1) / scale;

  Common::CameraBase::Ptr pCamera = refFrame->getCameraModel();
  Eigen::Vector2f *pImgGradientVec = refFrame->getPyramidImageGradientVec(level);
  if (nullptr == pImgGradientVec) {
    throw std::runtime_error("Image gradient not present.");
  }

  // XXX: This is NOT a depth map in the traditional sense!!! It's a list of hypotheses.
  Common::DepthHypothesisBase *pDepthMap = refFrame->getFeatureDescriptors(level);
  int nDepthMapSize = refFrame->getFeatureSize(level);
  if (pDepthMap == NULL) {
    printf("[DirImgAlignCPU]: no depth map available for image alignment...\n");
    throw std::runtime_error("No depth map.");
  }

  Eigen::Matrix<float, 3, 6> SE3Jacobian = Eigen::Matrix<float, 3, 6>::Zero();
  SE3Jacobian(0, 3) = 1;
  SE3Jacobian(1, 4) = 1;
  SE3Jacobian(2, 5) = 1;

  cout << "Computing over " << nDepthMapSize << " steps..." << endl;
  for (int i = 0; i < nDepthMapSize; i++) {
    Common::DepthHypothesisBase pixelDepth = pDepthMap[i];
    if (!pixelDepth.bValidated) {
      cerr << "Skipping invalid pixel; should not hapen in DynSLAM" << endl;
      continue;
    }

    int r = pixelDepth.pixel(0);
    int c = pixelDepth.pixel(1);
    int index = r * nCols + c;
    Eigen::Vector3f ray = pixelDepth.unitRay;
    Eigen::Vector3f P3D = ray * pixelDepth.rayDepth;
    Eigen::Vector2f pixelGradient = pImgGradientVec[index];

    Eigen::Matrix<float, 2, 3> projectionJacobian;
    pCamera->projectionJacobian(P3D, level, projectionJacobian);

    SE3Jacobian(0, 1) = P3D(2);
    SE3Jacobian(0, 2) = P3D(1) * (-1.0f);
    SE3Jacobian(1, 0) = P3D(2) * (-1.0f);
    SE3Jacobian(1, 2) = P3D(0);
    SE3Jacobian(2, 0) = P3D(1);
    SE3Jacobian(2, 1) = P3D(0) * (-1.0f);

    Eigen::Matrix<float, 1, 6> J = pixelGradient.transpose() * projectionJacobian * SE3Jacobian;
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> H = J.transpose() * J;

    memcpy(mPreCompJacobian + 6 * i, J.data(), 6 * sizeof(float));
    memcpy(mPreCompHessian + 36 * i, H.data(), 36 * sizeof(float));
  }
}

void DirImgAlignCPU::solverGaussNewton(const T_FramePtr &refFrame,
                                       const T_FramePtr &curFrame,
                                       int level,
                                       Eigen::Matrix4f &Tref2cur) {
  float prevCost = std::numeric_limits<float>::max();

  Eigen::Matrix4f T_prev = Tref2cur;
  Eigen::Matrix4f T_cur = Tref2cur;

  for (int i = 0; i < mnMaxIterations; i++) {
    Eigen::Matrix<float, 6, 1> epsilon;

// 	  Common::T_time tStart = Common::currentTime();
// 	  float cost = gaussNewtonUpdateStepSSE(refFrame, curFrame, level, T_cur, epsilon);
    float cost = gaussNewtonUpdateStep(refFrame, curFrame, level, T_cur, epsilon);
// 	  printf("GN time: %lld\n", Common::elapsedTime(tStart, Common::US));

    printf("GN cost: %10.6f | Epsilon: %f, %f, %f, %f, %f, %f\n", cost,
           epsilon(0), epsilon(1), epsilon(2), epsilon(3), epsilon(4), epsilon(5));

    bool stop = std::isnan(epsilon(0));
    if (stop) {
      std::cout << "epsilon(0) is NaN | Stopping.." << std::endl;
    }
    if(cost > prevCost) {
      std::cout << "Cost increased from previous iteration. Was " << prevCost << " and it's now "
                << cost << ". Stopping." << std::endl;
    }
    if (stop || cost > prevCost) {
      T_cur = T_prev;
      break;
    }
    T_prev = T_cur;

    Sophus::SE3<float>::Tangent tangent;
    tangent << epsilon(3), epsilon(4), epsilon(5), epsilon(0), epsilon(1), epsilon(2);
    Sophus::SE3<float> ExpEpsilon = Sophus::SE3<float>::exp(tangent);
    Sophus::SE3<float> deltaT = ExpEpsilon.inverse();

    // Update our current estimate
    T_cur = T_cur * (deltaT.matrix());

    prevCost = cost;
    if (epsilon.cwiseAbs().maxCoeff() < mConvergenceEps) {
      std::cout << "Epsilon max coef is too small. Stopping Gauss-Newton after " << i
                << " iterations." << std::endl;
      break;
    }

    if ((i + 1) == mnMaxIterations) {
      std::cerr << "Warning: direct alignment reached maximum number of iterations and stopped."
                << std::endl;
    }
  }


  Tref2cur = T_cur;
}

float DirImgAlignCPU::gaussNewtonUpdateStep(const T_FramePtr &refFrame,
                                            const T_FramePtr &curFrame,
                                            int level,
                                            Eigen::Matrix4f T_ref2cur,
                                            Eigen::Matrix<float, 6, 1> &outEpsilon) {
  using namespace std;
  int scale = 1 << level;
  int nRows = refFrame->getFrameSize()(0) / scale;
  int nCols = refFrame->getFrameSize()(1) / scale;
  int weakGradientCount = 0;

  /// XXX: does this help? Seems like it doesn't.
  float learningRate = 1.0f;

  Eigen::Vector2f *pGradient = refFrame->getPyramidImageGradientVec(level);
  float *pGradientMag = refFrame->getPyramidImageGradientMag(level);
  Common::DepthHypothesisBase *pDepthMap = refFrame->getFeatureDescriptors(level);
  Common::CameraBase::Ptr pCurCamera = refFrame->getCameraModel();
  int nDepthMapSize = refFrame->getFeatureSize(level);

  unsigned char *pRefImageData = refFrame->getPyramidImage(level);
  unsigned char *pCurImageData = curFrame->getPyramidImage(level);

//  cv::Mat1b preview(nRows, nCols, pCurImageData);
//  cv::Mat1b prev2(nRows, nCols, pRefImageData);
//  cv::imshow("Current image", preview);
//  cv::imshow("Reference image", prev2);
//  cv::waitKey();

  Eigen::Matrix<float, 6, 1> Jsum = Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6, Eigen::RowMajor>
      Hsum = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();

  float totalCost = 0.0f;

  float sum_wIrIk = 0.0;
  float sum_wIr = 0.0;
  float sum_wIrIr = 0.0;
  float sum_wIk = 0.0;
  float sum_w = 0.0;

  int nCount = 0;
  nDepthMapSize = nDepthMapSize / 4 * 4;

  for (int i = 0; i < nDepthMapSize; i++) {
    Common::DepthHypothesisBase pixelDepth = pDepthMap[i];
    if (pixelDepth.bNoGoodMatching || pixelDepth.bDiverged) {
      continue;
    }

    int r = pixelDepth.pixel(0);
    int c = pixelDepth.pixel(1);
    int index = r * nCols + c;

//    Eigen::Vector2f &gradient = pGradient[index];
    float gradientMag = pGradientMag[index];
    if (gradientMag < mMinGradMagnitude) {
      weakGradientCount++;
      continue;
    }

    // get point 3D position
    Eigen::Vector3f ray = pixelDepth.unitRay;
    Eigen::Vector3f P3D_refF = ray * pixelDepth.rayDepth;
    Eigen::Vector3f P3D_curF = T_ref2cur.block<3, 3>(0, 0) * P3D_refF + T_ref2cur.block<3, 1>(0, 3);

    // project it to current image
    Eigen::Vector2f pixel_cur;
    if (!pCurCamera->project(P3D_curF, pixel_cur)) {
//      cout << "pCurCamera->project failed to project point." << endl;
//      cout << "Ray: " << ray << " | Ref. frame 3D point: " << P3D_refF << " | Curr. frame 3D point: " << P3D_curF << endl;
//      cout << "Ray depth was: " << pixelDepth.rayDepth << endl;
      continue;
    }

    // Not using masking in dynslam at the moment.
//    if(!curFrame->pixelLieOutsideImageMask(pixel_cur(1), pixel_cur(0))) {
//      cout << "Pixel outside image..." << ray << " | " << P3D_refF << " | " << P3D_curF << endl;
//      cout << "Ray depth was: " << pixelDepth.rayDepth << endl;
//      continue;
//    }
    pixel_cur /= (scale * 1.0f);

    float residual0 = 0.0f;
    float intensity_ref = 0.0;
    float intensity_cur = 0.0;

    // get interpolated pixel intensity
    intensity_ref = pixelDepth.intensity; // static_cast<float>(pRefImageData[index]);
    intensity_cur =
        Common::bilinearInterpolation(pCurImageData, nRows, nCols, pixel_cur(1), pixel_cur(0));
    residual0 = mAffineBrightness_a * intensity_ref + mAffineBrightness_b - intensity_cur;

    if (std::isnan(residual0)) {
      std::cerr << "residual0 NaN! " << std::endl;
    }

    // get robust weighting
    float weight = mRobustLossFunction->getWeight(residual0);

    if (i % 555 == 0) {
//      cout << "Residual sample: " << residual0 << " | Weight: " << weight << endl;
//      cout << "Whereby the reference intensity is: " << intensity_ref << ", and the current one is:"
//           << intensity_cur << ". Affine brightness params (a,b) = " << mAffineBrightness_a
//           << " and " << mAffineBrightness_b << "."<< endl;
//      cout << "Current intensity sampled  and interp'd bilinearly from coords: x = " << pixel_cur(0)
//           << ", y = " << pixel_cur(1) << endl;
//      cout << "Coordinates in reference frame: row = " << pixelDepth.pixel(0) << ", col = "
//           << pixelDepth.pixel(1) << "." << endl;
////      cout << "Ray: " << ray << " | 3D reference: " << P3D_refF << " | 3D current: " << P3D_curF << endl;
    }

    if (std::isnan(weight)) {
      std::cerr << "Weight became NaN!" << std::endl;
    }

    totalCost += weight * residual0 * residual0;
    nCount++;

    // compute hessian, jacobian
    Eigen::Matrix<float, 6, 1> JiT;
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> Hi;

    memcpy(JiT.data(), mPreCompJacobian + 6 * i, 6 * sizeof(float));
    memcpy(Hi.data(), mPreCompHessian + 36 * i, 36 * sizeof(float));

    // accumulate hessian and jacobian
    JiT = weight * residual0 * JiT;
    Hi = mAffineBrightness_a * weight * Hi;

    // Can parallelize trivially everything up to here, then reduce the rest of the stuff.

    float weightByVariance = 1.0f; //std::min(0.0001f / invDepth.variance, 100.0f);
    Jsum += (JiT * weightByVariance);
    Hsum += (Hi * weightByVariance);

    // accumulate variables for affine lighting model
    sum_wIrIk += (weight * intensity_ref * intensity_cur);
    sum_wIr += (weight * intensity_ref);
    sum_wIrIr += (weight * intensity_ref * intensity_ref);
    sum_wIk += (weight * intensity_cur);
    sum_w += weight;
  }

  // update affine brightness model
  float prev_affine_a = mAffineBrightness_a;
  float prev_affine_b = mAffineBrightness_b;

  mAffineBrightness_a = (sum_wIrIk - prev_affine_b * sum_wIr) / sum_wIrIr;
  mAffineBrightness_b = (sum_wIk - prev_affine_a * sum_wIr) / sum_w;

  if (nCount <= 0) {
    std::cerr << "ERROR: nCount <= 0 | nCount == " << nCount << std::endl;
  }
  else if (nCount < 500) {
    std::cerr << "WARNING: few points used for actual optimization: " << nCount << std::endl;
  }

  std::cout << "Dropped " << weakGradientCount << " terms due to weak gradient." << std::endl;

  // update epsilon
  outEpsilon = learningRate * (-1.0f) * Hsum.ldlt().solve(Jsum);
  return sqrt(totalCost / (nCount * 1.0f));
}

// Faster version of the above method; looks like it's still under construction.
// Is this faster than Eigen's built-in vectorization?
/*
float DirImgAlignCPU::gaussNewtonUpdateStepSSE(const T_FramePtr &refFrame,
                                               const T_FramePtr &curFrame,
                                               int level,
                                               Eigen::Matrix4f T_ref2cur,
                                               Eigen::Matrix<float, 6, 1> &outEpsilon) {
  int scale = 1 << level;
  int nRows = refFrame->getFrameSize()(0) / scale;
  int nCols = refFrame->getFrameSize()(1) / scale;

  Eigen::Vector2f *pGradient = refFrame->getPyramidImageGradientVec(level);
  Common::DepthHypothesisBase *pDepthMap = refFrame->getFeatureDescriptors(level);
  Common::CameraBase::Ptr pCurCamera = refFrame->getCameraModel();
  int nDepthMapSize = refFrame->getFeatureSize(level);

  unsigned char *pRefImageData = refFrame->getPyramidImage(level);
  unsigned char *pCurImageData = curFrame->getPyramidImage(level);

  Eigen::Matrix<float, 6, 1> Jsum = Eigen::Matrix<float, 6, 1>::Zero();
  Eigen::Matrix<float, 6, 6, Eigen::RowMajor>
      Hsum = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();

  int nCount = 0;

  const __m128i sse_nCols = _mm_set_epi32(nCols, nCols, nCols, nCols);
  Common::SSE_m128_m44 sse_T_ref2cur;
  sse_T_ref2cur.row[0] =
      _mm_set_ps(T_ref2cur(0, 3), T_ref2cur(0, 2), T_ref2cur(0, 1), T_ref2cur(0, 0));
  sse_T_ref2cur.row[1] =
      _mm_set_ps(T_ref2cur(1, 3), T_ref2cur(1, 2), T_ref2cur(1, 1), T_ref2cur(1, 0));
  sse_T_ref2cur.row[2] =
      _mm_set_ps(T_ref2cur(2, 3), T_ref2cur(2, 2), T_ref2cur(2, 1), T_ref2cur(2, 0));
  sse_T_ref2cur.row[3] = _mm_set_ps(1.0, 0.0, 0.0, 0.0);

  __m128 sse_totalCost = _mm_set1_ps(0.0);
  __m128 sse_sum_wIrIk = sse_totalCost;
  __m128 sse_sum_wIr = sse_totalCost;
  __m128 sse_sum_wIrIr = sse_totalCost;
  __m128 sse_sum_wIk = sse_totalCost;
  __m128 sse_sum_w = sse_totalCost;

  for (int i = 0; i < nDepthMapSize / 4; i++) {
    // get depth map data
    int i4 = 4 * i;
    Common::DepthHypothesisBase pixelDepth_0 = pDepthMap[i4];
    Common::DepthHypothesisBase pixelDepth_1 = pDepthMap[i4 + 1];
    Common::DepthHypothesisBase pixelDepth_2 = pDepthMap[i4 + 2];
    Common::DepthHypothesisBase pixelDepth_3 = pDepthMap[i4 + 3];

    // get extract all required data
    Eigen::Vector3f ray[4];
    ray[0] = pixelDepth_0.unitRay;
    ray[1] = pixelDepth_1.unitRay;
    ray[2] = pixelDepth_2.unitRay;
    ray[3] = pixelDepth_3.unitRay;

    float rayDepth[4];
    rayDepth[0] = pixelDepth_0.rayDepth;
    rayDepth[1] = pixelDepth_1.rayDepth;
    rayDepth[2] = pixelDepth_2.rayDepth;
    rayDepth[3] = pixelDepth_3.rayDepth;

    float intensity_ref[4];
    intensity_ref[0] = pixelDepth_0.intensity;
    intensity_ref[1] = pixelDepth_1.intensity;
    intensity_ref[2] = pixelDepth_2.intensity;
    intensity_ref[3] = pixelDepth_3.intensity;

    // TODO: add depth mask

    // get point 3D position
    __m128 sse_ray_x = _mm_set_ps(ray[3](0), ray[2](0), ray[1](0), ray[0](0));
    __m128 sse_ray_y = _mm_set_ps(ray[3](1), ray[2](1), ray[1](1), ray[0](1));
    __m128 sse_ray_z = _mm_set_ps(ray[3](2), ray[2](2), ray[1](2), ray[0](2));

    __m128 sse_rayDepth = _mm_load_ps(rayDepth);

    __m128 sse_p3d_refF_x = _mm_mul_ps(sse_ray_x, sse_rayDepth);
    __m128 sse_p3d_refF_y = _mm_mul_ps(sse_ray_y, sse_rayDepth);
    __m128 sse_p3d_refF_z = _mm_mul_ps(sse_ray_z, sse_rayDepth);

    __m128 sse_ones = _mm_set1_ps(1.0f);
    _MM_TRANSPOSE4_PS(sse_p3d_refF_x, sse_p3d_refF_y, sse_p3d_refF_z, sse_ones);
    __m128 sse_p3d_refF_0 = sse_p3d_refF_x;
    __m128 sse_p3d_refF_1 = sse_p3d_refF_y;
    __m128 sse_p3d_refF_2 = sse_p3d_refF_z;
    __m128 sse_p3d_refF_3 = sse_ones;

    __m128 sse_p3d_curF_0 = sse_T_ref2cur.mul(sse_p3d_refF_0);
    __m128 sse_p3d_curF_1 = sse_T_ref2cur.mul(sse_p3d_refF_1);
    __m128 sse_p3d_curF_2 = sse_T_ref2cur.mul(sse_p3d_refF_2);
    __m128 sse_p3d_curF_3 = sse_T_ref2cur.mul(sse_p3d_refF_3);

    // project it to current image
    _MM_TRANSPOSE4_PS(sse_p3d_curF_0, sse_p3d_curF_1, sse_p3d_curF_2, sse_p3d_curF_3);
    Common::SSE_m128_v3 sse_scenePointsV3;
    sse_scenePointsV3.m[0] = sse_p3d_curF_0;
    sse_scenePointsV3.m[1] = sse_p3d_curF_1;
    sse_scenePointsV3.m[2] = sse_p3d_curF_2;
    Common::SSE_m128_v2 sse_pixel_cur = pCurCamera->project(sse_scenePointsV3);
    sse_pixel_cur = sse_pixel_cur.div(scale);

    // get interpolated pixel intensity
    __m128 sse_intensity_ref = _mm_load_ps(intensity_ref);
    __m128 sse_intensity_cur =
        sse_intensity_ref; //Common::bilinearInterpolation(pCurImageData, nRows, nCols, sse_pixel_cur);

    // get robust weighting
    __m128 sse_residual0 =
        _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(mAffineBrightness_a), sse_intensity_ref),
                              _mm_set1_ps(mAffineBrightness_b)), sse_intensity_cur);
    __m128 sse_weight = mRobustLossFunction->getWeight(sse_residual0);
    __m128 sse_weightResidual = _mm_mul_ps(sse_weight, sse_residual0);
    sse_totalCost = _mm_add_ps(sse_totalCost, _mm_mul_ps(sse_weightResidual, sse_residual0));
    nCount++;

    // compute hessian, jacobian
    Eigen::Matrix<float, 6, 1> JiT[4];
    Eigen::Matrix<float, 6, 6, Eigen::RowMajor> Hi[4];

    memcpy(JiT[0].data(), mPreCompJacobian + 6 * i4, 6 * sizeof(float));
    memcpy(Hi[0].data(), mPreCompHessian + 36 * i4, 36 * sizeof(float));

    memcpy(JiT[1].data(), mPreCompJacobian + 6 * (i4 + 1), 6 * sizeof(float));
    memcpy(Hi[1].data(), mPreCompHessian + 36 * (i4 + 1), 36 * sizeof(float));

    memcpy(JiT[2].data(), mPreCompJacobian + 6 * (i4 + 2), 6 * sizeof(float));
    memcpy(Hi[2].data(), mPreCompHessian + 36 * (i4 + 2), 36 * sizeof(float));

    memcpy(JiT[3].data(), mPreCompJacobian + 6 * (i4 + 3), 6 * sizeof(float));
    memcpy(Hi[3].data(), mPreCompHessian + 36 * (i4 + 3), 36 * sizeof(float));

    // accumulate hessian and jacobian
    float weightResidual[4], weight[4];
    _mm_store_ps(weightResidual, sse_weightResidual);
    _mm_store_ps(weight, sse_weight);

    JiT[0] = weightResidual[0] * JiT[0];
    JiT[1] = weightResidual[1] * JiT[1];
    JiT[2] = weightResidual[2] * JiT[2];
    JiT[3] = weightResidual[3] * JiT[3];

    Hi[0] = mAffineBrightness_a * weight[0] * Hi[0];
    Hi[1] = mAffineBrightness_a * weight[1] * Hi[1];
    Hi[2] = mAffineBrightness_a * weight[2] * Hi[2];
    Hi[3] = mAffineBrightness_a * weight[3] * Hi[3];

    Jsum += (JiT[0] + JiT[1] + JiT[2] + JiT[3]);
    Hsum += (Hi[0] + Hi[1] + Hi[2] + Hi[3]);

    // accumulate variables for affine lighting model
    __m128 sse_weightIntensityRef = _mm_mul_ps(sse_weight, sse_intensity_ref);
    sse_sum_wIrIk =
        _mm_add_ps(sse_sum_wIrIk, _mm_mul_ps(sse_weightIntensityRef, sse_intensity_cur));
    sse_sum_wIr = _mm_add_ps(sse_sum_wIr, sse_weightIntensityRef);
    sse_sum_wIrIr =
        _mm_add_ps(sse_sum_wIrIr, _mm_mul_ps(sse_weightIntensityRef, sse_intensity_ref));
    sse_sum_wIk = _mm_add_ps(sse_sum_wIk, _mm_mul_ps(sse_weight, sse_intensity_cur));
    sse_sum_w = _mm_add_ps(sse_sum_w, sse_weight);
  }

  // update affine brightness model
  float prev_affine_a = mAffineBrightness_a;
  float prev_affine_b = mAffineBrightness_b;

  float sum_wIrIk[4], sum_wIr[4], sum_wIrIr[4], sum_wIk[4], sum_w[4];
  _mm_store_ps(sum_wIrIk, sse_sum_wIrIk);
  _mm_store_ps(sum_wIr, sse_sum_wIr);
  _mm_store_ps(sum_wIrIr, sse_sum_wIrIr);
  _mm_store_ps(sum_wIk, sse_sum_wIk);
  _mm_store_ps(sum_w, sse_sum_w);

  float _sum_wIrIk = ACCUMULATE_SUM_V4(sum_wIrIk);
  float _sum_wIr = ACCUMULATE_SUM_V4(sum_wIr);
  float _sum_wIrIr = ACCUMULATE_SUM_V4(sum_wIrIr);
  float _sum_wIk = ACCUMULATE_SUM_V4(sum_wIk);
  float _sum_w = ACCUMULATE_SUM_V4(sum_w);

  mAffineBrightness_a = (_sum_wIrIk - prev_affine_b * _sum_wIr) / _sum_wIrIr;
  mAffineBrightness_b = (_sum_wIk - prev_affine_a * _sum_wIr) / _sum_w;

  // update epsilon
  outEpsilon = (-1.0f) * Hsum.ldlt().solve(Jsum);
  float totalCost[4];
  _mm_store_ps(totalCost, sse_totalCost);
  float _totalCost = ACCUMULATE_SUM_V4(totalCost);
  return sqrt(_totalCost / (nCount * 1.0f));
}
 */

}
}