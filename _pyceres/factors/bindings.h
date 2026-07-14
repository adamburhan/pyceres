#pragma once

#include "_pyceres/helpers.h"

#include <ceres/ceres.h>
#include <ceres/normal_prior.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline Eigen::MatrixXd SqrtInformation(const Eigen::MatrixXd& covariance) {
  return covariance.inverse().llt().matrixL().transpose();
}

// Mahalanobis squared distance between two parameters.
class NormalError {
 public:
  explicit NormalError(const Eigen::MatrixXd& covariance)
      : sqrt_information_(SqrtInformation(covariance)) {
    THROW_CHECK_EQ(covariance.rows(), covariance.cols());
  }

  static ceres::CostFunction* Create(const Eigen::MatrixXd& covariance) {
    auto* cost_function = new ceres::DynamicAutoDiffCostFunction<NormalError>(
        new NormalError(covariance));
    const int dimension = covariance.rows();
    cost_function->AddParameterBlock(dimension);
    cost_function->AddParameterBlock(dimension);
    cost_function->SetNumResiduals(dimension);
    return cost_function;
  }

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals_ptr) const {
    const int dimension = sqrt_information_.rows();
    for (int i = 0; i < dimension; ++i) {
      residuals_ptr[i] = parameters[0][i] - parameters[1][i];
    }
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> residuals(residuals_ptr,
                                                              dimension);
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }

 private:
  const Eigen::MatrixXd sqrt_information_;
};


/*
Depth factors
*/


// Domains in which a monocular depth residual can be expressed. Transform()
// returns false if the depth is invalid in the domain (non-positive).
struct DepthDomain {
  template <typename T>
  static bool Transform(const T& z, T* h) {
    *h = z;
    return true;
  }
};

struct InverseDepthDomain {
  template <typename T>
  static bool Transform(const T& z, T* h) {
    if (z <= T(0.0)) {
      return false;
    }
    *h = T(1.0) / z;
    return true;
  }
};

struct LogDepthDomain {
  template <typename T>
  static bool Transform(const T& z, T* h) {
    using std::log;
    if (z <= T(0.0)) {
      return false;
    }
    *h = log(z);
    return true;
  }
};

// Monocular depth factor with a per-image affine correction:
//   r = (h(alpha * depth + beta) - h(z_cam)) / stddev
// where h is the domain transform and z_cam the third component of the point
// in the camera frame. Parameter blocks: cam_from_world (7: quat x, y, z, w,
// then translation, matching COLMAP's Rigid3d; use
// ProductManifold(EigenQuaternionManifold, EuclideanManifold(3))),
// point3D (world), alpha, beta.
template <typename Domain>
class MonoDepthError {
 public:
  MonoDepthError(const double depth, const double stddev)
      : depth_(depth), inv_stddev_(1.0 / stddev) {
    THROW_CHECK_GT(stddev, 0.0);
  }

  static ceres::CostFunction* Create(const double depth, const double stddev) {
    return new ceres::
        AutoDiffCostFunction<MonoDepthError<Domain>, 1, 7, 3, 1, 1>(
            new MonoDepthError<Domain>(depth, stddev));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const point3D,
                  const T* const alpha,
                  const T* const beta,
                  T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q_cam_from_world(
        cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam_from_world(
        cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> point3D_world(point3D);
    const T z_cam = (q_cam_from_world * point3D_world + t_cam_from_world)(2);

    T h_measured, h_projected;
    if (!Domain::Transform(alpha[0] * T(depth_) + beta[0], &h_measured) ||
        !Domain::Transform(z_cam, &h_projected)) {
      return false;
    }
    residuals[0] = T(inv_stddev_) * (h_measured - h_projected);
    return true;
  }

 private:
  const double depth_;
  const double inv_stddev_;
};

// Max-mixture (Olson & Agarwal) version of MonoDepthError with k modes
// (depth_k, stddev_k, weight_k) sharing the affine correction. The mode is
// selected by minimizing the whitened squared error plus the log-normalizer
// 2 * log(stddev_k / weight_k); the residual is the winning mode's whitened
// error, so the normalizers affect selection only (the reported objective is
// discontinuous at mode switches, but the linearization stays Gaussian).
// Same parameter blocks as MonoDepthError.
template <typename Domain>
class MonoDepthMaxMixError {
 public:
  MonoDepthMaxMixError(const Eigen::VectorXd& depths,
                       const Eigen::VectorXd& stddevs,
                       const Eigen::VectorXd& weights)
      : depths_(depths), inv_stddevs_(stddevs.cwiseInverse()) {
    THROW_CHECK_GT(depths.size(), 0);
    THROW_CHECK_EQ(depths.size(), stddevs.size());
    THROW_CHECK_EQ(depths.size(), weights.size());
    THROW_CHECK((stddevs.array() > 0.0).all());
    THROW_CHECK((weights.array() > 0.0).all());
    log_terms_ = 2.0 * (stddevs.array().log() - weights.array().log());
  }

  static ceres::CostFunction* Create(const Eigen::VectorXd& depths,
                                     const Eigen::VectorXd& stddevs,
                                     const Eigen::VectorXd& weights) {
    return new ceres::
        AutoDiffCostFunction<MonoDepthMaxMixError<Domain>, 1, 7, 3, 1, 1>(
            new MonoDepthMaxMixError<Domain>(depths, stddevs, weights));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const point3D,
                  const T* const alpha,
                  const T* const beta,
                  T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q_cam_from_world(
        cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cam_from_world(
        cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> point3D_world(point3D);
    const T z_cam = (q_cam_from_world * point3D_world + t_cam_from_world)(2);

    T h_projected;
    if (!Domain::Transform(z_cam, &h_projected)) {
      return false;
    }
    bool any_valid = false;
    T best_cost;
    T best_whitened;
    for (int k = 0; k < depths_.size(); ++k) {
      T h_measured;
      // Modes with an invalid corrected depth have zero likelihood.
      if (!Domain::Transform(alpha[0] * T(depths_[k]) + beta[0],
                             &h_measured)) {
        continue;
      }
      const T whitened = T(inv_stddevs_[k]) * (h_measured - h_projected);
      const T cost = whitened * whitened + T(log_terms_[k]);
      if (!any_valid || cost < best_cost) {
        best_cost = cost;
        best_whitened = whitened;
        any_valid = true;
      }
    }
    if (!any_valid) {
      return false;
    }
    residuals[0] = best_whitened;
    return true;
  }

 private:
  const Eigen::VectorXd depths_;
  const Eigen::VectorXd inv_stddevs_;
  Eigen::ArrayXd log_terms_;
};

void BindFactors(py::module& m) {
  m.def(
      "NormalPrior",
      [](const Eigen::VectorXd& mean,
         const Eigen::MatrixXd& covariance) -> ceres::CostFunction* {
        THROW_CHECK_EQ(covariance.cols(), mean.size());
        THROW_CHECK_EQ(covariance.cols(), covariance.rows());
        return new ceres::NormalPrior(SqrtInformation(covariance), mean);
      },
      py::arg("mean"),
      py::arg("covariance"));

  m.def("NormalError", &NormalError::Create, py::arg("covariance"));

  m.def("DepthError",
        &MonoDepthError<DepthDomain>::Create,
        py::arg("depth"),
        py::arg("stddev"));
  m.def("InvDepthError",
        &MonoDepthError<InverseDepthDomain>::Create,
        py::arg("depth"),
        py::arg("stddev"));
  m.def("LogDepthError",
        &MonoDepthError<LogDepthDomain>::Create,
        py::arg("depth"),
        py::arg("stddev"));

  m.def("DepthErrorMaxMix",
        &MonoDepthMaxMixError<DepthDomain>::Create,
        py::arg("depths"),
        py::arg("stddevs"),
        py::arg("weights"));
  m.def("InvDepthErrorMaxMix",
        &MonoDepthMaxMixError<InverseDepthDomain>::Create,
        py::arg("depths"),
        py::arg("stddevs"),
        py::arg("weights"));
  m.def("LogDepthErrorMaxMix",
        &MonoDepthMaxMixError<LogDepthDomain>::Create,
        py::arg("depths"),
        py::arg("stddevs"),
        py::arg("weights"));
}
