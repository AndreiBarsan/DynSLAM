#include "transformation.h"
#include "../helperFunctions.hpp"

namespace VGUGV
{
  namespace Common
  {
    Transformation::Transformation() 
    { 
      mT = Eigen::Matrix4f::Identity();
      mTinv = mT;
    }
    
    Transformation::Transformation(Eigen::Matrix4f T)
    : mT(T)
    , mTinv(mT.inverse()){};
    
    Transformation::Transformation(Eigen::Vector3f euler, Eigen::Vector3f translation)
    {
      Eigen::Matrix3f _R;
      Common::euler2rotMatrix<float>(euler(0), euler(1), euler(2),
				     _R(0,0), _R(0,1), _R(0,2),
				     _R(1,0), _R(1,1), _R(1,2),
				     _R(2,0), _R(2,1), _R(2,2));
      Eigen::Vector3f _t = -1.0f * _R.transpose() * translation;
      
      mT = Eigen::Matrix4f::Identity();
      mT.block<3,3>(0,0) = _R;
      mT.block<3,1>(0,3) = _t;
      
      mTinv = mT.inverse();
    }
    
    void Transformation::setT(Eigen::Matrix4f T)
    {
      mT = T;
      mTinv = mT.inverse();
    }
    
    Eigen::Matrix4f Transformation::getTMatrix()
    {
      return mT;
    }
    
    Eigen::Matrix4f Transformation::getTMatrixInv()
    {
      return mTinv;
    }
    
    float Transformation::angularDistance(Transformation T)
    {
      Eigen::Matrix4f Tmatrix = T.getTMatrix();
      Eigen::Matrix4f difference = mTinv * Tmatrix;
      Eigen::Matrix3f R = difference.block<3, 3>(0, 0);
      Eigen::Vector3f eulerAngles = R.eulerAngles(2, 1, 0);
      float phi, theta, psi;
      rotMatrix2Euler<float>(R(0, 0), R(0, 1), R(0, 2),
			     R(1, 0), R(1, 1), R(1, 2),
			     R(2, 0), R(2, 1), R(2, 2),
			     phi, theta, psi);
      return sqrt(phi * phi + theta * theta + psi * psi);
    }
    
    float Transformation::translationDistance(Transformation T)
    {
      Eigen::Matrix4f Tmatrix = T.getTMatrix();
      Eigen::Matrix4f difference = mTinv * Tmatrix;
      Eigen::Vector3f translation = difference.block<3, 1>(0, 3);
      return translation.norm();
    }
    
    Eigen::Vector3f Transformation::getTranslation()
    {
      return -1.0 * mT.block<3,3>(0,0) * mT.block<3,1>(0,3);
    }
    
    Eigen::Vector3f Transformation::getEulerAngle()
    {
      Eigen::Matrix3f R = mT.block<3,3>(0,0);
      float phi, theta, psi;
      VGUGV::Common::rotMatrix2Euler<float>(R(0,0), R(0,1), R(0,2),
					    R(1,0), R(1,1), R(1,2),
					    R(2,0), R(2,1), R(2,2),
					    phi, theta, psi);
      return Eigen::Vector3f(phi, theta, psi);
    }
    
    Transformation Transformation::invMul(Transformation T)
    { 
      return Transformation(mTinv * T.getTMatrix());
    }
    
    Transformation Transformation::mulInv(Transformation T)
    {
      return Transformation(mT * T.getTMatrixInv());
    }
    
    Transformation Transformation::mul(Transformation T)
    {
      return Transformation(mT * T.getTMatrix());
    }
    
  }
}