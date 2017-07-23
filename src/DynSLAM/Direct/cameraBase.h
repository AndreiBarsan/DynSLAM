#ifndef _VGUGV_COMMON_CAMERABASE_
#define _VGUGV_COMMON_CAMERABASE_

#include <memory>
#include <Eigen/Dense>
#include <vector>
#include "cudaDefs.h"
#include "commDefs.h"
#include "math/Vector.h"
#include <pmmintrin.h>

namespace VGUGV
{
	namespace Common
	{
		class CameraBase{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		public:
			typedef std::shared_ptr<CameraBase> Ptr;

			// constructors
			virtual ~CameraBase() {};

		public:
			virtual bool project(const Eigen::Vector3f& scenePoint, Eigen::Vector2f& pixelPoint, float* jacobians = NULL) = 0;
			virtual bool backProject(const Eigen::Vector2f& pixelPoint, Eigen::Vector3f& ray) = 0;
			virtual bool backProject(int r, int c, Eigen::Vector3f& ray) = 0;
			virtual bool projectionJacobian(const Eigen::Vector3f& scenePoint, int pyramidLevel, Eigen::Matrix<float, 2, 3>& jacobian) = 0;

			// retrieve
			virtual Eigen::Vector2i getCameraSize() = 0;
			virtual void getK(Eigen::Matrix3f& K) = 0;
			virtual void getK(float& fx, float& fy, float& cx, float& cy) = 0;
			virtual void getKinv(Eigen::Matrix3f& Kinv) = 0;
			virtual void getKinv(float& fxInv, float& fyInv, float& cxInv, float& cyInv) = 0;
			virtual void getDistoritionParams(std::vector<float>&) = 0;
			virtual void getAdditionalParams(std::vector<float>&) = 0;
			virtual Vector3<float>* getRayPtrs(Common::DEVICE_TYPE device) = 0;
			
			// set 
			virtual void setDistortionParams(std::vector<float>) = 0;
			
		public: // sse related functions
		  //! Constructor for base Frame class
		  /*!
		   *  \param scenePoints 3 __m128 array, each array consists of x, y, z values of 4 scene points respectively 
		   *  \return 2 __m128 array, each array consists of x, y pixel coordinates of 4 scene points respectively
		   */
		  virtual SSE_m128_v2 project(SSE_m128_v3 scenePoints) = 0;
		};
	}
}

#endif