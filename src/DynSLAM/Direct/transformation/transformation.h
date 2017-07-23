#ifndef _VGUGV_COMMON_POSE_
#define _VGUGV_COMMON_POSE_

#include <Eigen/Dense>
#include <memory>

namespace VGUGV
{
  namespace Common
  {
    class Transformation
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      typedef std::shared_ptr<Transformation> Ptr;      
    public: 
      Transformation();       
      Transformation(Eigen::Matrix4f T);      
      Transformation(Eigen::Vector3f euler, Eigen::Vector3f translation);
      
    public:
      void setT(Eigen::Matrix4f T);
      Eigen::Matrix4f getTMatrix();
      Eigen::Matrix4f getTMatrixInv();
      float angularDistance(Transformation T);
      float translationDistance(Transformation T);
      
    public:
      Eigen::Vector3f getTranslation();
      Eigen::Vector3f getEulerAngle();
      
    public:
      Transformation invMul(Transformation T);
      Transformation mulInv(Transformation T);
      Transformation mul(Transformation T);
      
    private:
      Eigen::Matrix4f mT;
      Eigen::Matrix4f mTinv;  
    };
  }
}

#endif