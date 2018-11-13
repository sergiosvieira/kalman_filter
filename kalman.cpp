#include <iostream>
#include <stdexcept>

#include "kalman.h"

KalmanFilter::KalmanFilter(
    double dt,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P)
  : A(A), C(C), Q(Q), R(R), P0(P),
    m(C.rows()), n(A.rows()), dt(dt), initialized(false),
    I(n, n), x0(n * n), x1(n * n)
{
  I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::MatrixXd& x0) {
  this->x0 = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init() {
  x0.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::update(const Eigen::VectorXd& y)
{

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");
  // Project the state ahead
  x1 = A * x0;
  // Project the error covariance ahead
  P = A * P * A.transpose() + Q;
  // Compute the Kalman Gain
  K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
  // Update the estimate via y
  Eigen::MatrixXd c = C * x1;

  x1 += K * (y - c);
  // Update the error covariance
  P = (I - K * C) * P;
  x0 = x1;
  t += dt;
}

void KalmanFilter::update(const Eigen::MatrixXd& y, double dt, const Eigen::MatrixXd A) {

  this->A = A;
  this->dt = dt;
  update(y);
}
