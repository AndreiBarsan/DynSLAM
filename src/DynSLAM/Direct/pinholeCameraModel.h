#ifndef _VGUGV_COMMON_PINHOLE_CAMEAR_MODEL_
#define _VGUGV_COMMON_PINHOLE_CAMEAR_MODEL_

#include "cameraBase.h"
#include <memory>

using namespace VGUGV::Common;

namespace VGUGV
{
	namespace Common
	{
		class PinholeCameraModel : public CameraBase
		{
		public:
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			typedef std::shared_ptr<PinholeCameraModel> Ptr;

		public:
		        // image size (row, col)
			PinholeCameraModel(Eigen::Vector2i& imageSize, Eigen::Matrix3f& K)
				: mK(K)
				, mKinv(K.inverse())
				, mImageSize(imageSize)
				, mRays(NULL)
				, mRays_CUDA(NULL){};
			~PinholeCameraModel();

			bool project(const Eigen::Vector3f& scenePoint, Eigen::Vector2f& pixelPoint, float* jacobians = NULL);
			bool backProject(const Eigen::Vector2f& pixelPoint, Eigen::Vector3f& ray);
			bool backProject(int r, int c, Eigen::Vector3f& ray);
			bool projectionJacobian(const Eigen::Vector3f& scenePoint, int pyramidLevel, Eigen::Matrix<float, 2, 3>& jacobian);

			// retrieve
			Eigen::Vector2i getCameraSize();
			void getK(Eigen::Matrix3f& K);
			void getK(float& fx, float& fy, float& cx, float& cy);
			void getKinv(Eigen::Matrix3f& Kinv);
			void getKinv(float& fxInv, float& fyInv, float& cxInv, float& cyInv);
			void getDistoritionParams(std::vector<float>&);
			void getAdditionalParams(std::vector<float>&);
			Vector3<float>* getRayPtrs(Common::DEVICE_TYPE device);
			
			// set 
			void setDistortionParams(std::vector<float>);
		public:
		  SSE_m128_v2 project(SSE_m128_v3 scenePoints);

		private:
			Eigen::Matrix3f mK;
			Eigen::Matrix3f mKinv;
			Eigen::Vector2i mImageSize;
			Vector3<float>* mRays;
			Vector3<float>* mRays_CUDA;
		};
	}
}
#endif