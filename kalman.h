#include <Eigen/Dense>

#pragma once

/*!
 * \brief The KalmanFilter class
 */
class KalmanFilter
{
    /*!
     * \brief Matrices for computation
     */
    Eigen::MatrixXd A;
    Eigen::MatrixXd C;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd P;
    Eigen::MatrixXd K;
    Eigen::MatrixXd P0;
public:
  /**
  * \brief Create a Kalman filter with the specified matrices.
  *   A - System dynamics matrix
  *   C - Output matrix
  *   Q - Process noise covariance
  *   R - Measurement noise covariance
  *   P - Estimate error covariance
  */
  KalmanFilter(
      double dt,
      const Eigen::MatrixXd& A,
      const Eigen::MatrixXd& C,
      const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R,
      const Eigen::MatrixXd& P
  );
  /*!
   * \brief set System dynamics matrix
   * \param Eigen::MatrixXd
   */
  void setA(Eigen::MatrixXd& A_)
  {
    this->A = A_;
  }
  /*!
   * \brief set Output matrix
   * \param Eigen::MatrixXd
   */
  void setC(Eigen::MatrixXd& C_)
  {
    this->C = C_;
  }
  /*!
   * \brief set Process noise covariance
   * \param Eigen::MatrixXd
   */
  void setQ(Eigen::MatrixXd& Q_)
  {
    this->Q = Q_;
  }
  /*!
   * \brief set Measurement noise covariance
   * \param Eigen::MatrixXd
   */
  void setR(Eigen::MatrixXd& R_)
  {
    this->R = R_;
  }
  /*!
   * \brief set Estimate error covariance
   * \param Eigen::MatrixXd
   */
  void setP(Eigen::MatrixXd& P_)
  {
    this->P = P_;
  }

  /*!
  * \brief Create a blank estimator.
  */
  KalmanFilter();

  /*!
  * \brief Initialize the filter with initial states as zero.
  */
  void init();

  /*!
  * \brief Initialize the filter with a guess for initial states.
  */
  void init(double t0, const Eigen::MatrixXd& x0);

  /*!
  * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
  */
  void update(const Eigen::VectorXd &y);

  /*!
  * \brief Update the estimated state based on measured values, using the given time step and dynamics matrix.
  */
  void update(const Eigen::MatrixXd& y, double dt, const Eigen::MatrixXd A);

  /*!
  * \brief Return the current state and time.
  */
  Eigen::MatrixXd state() const
  {
      return x0;
  }
  double time() const
  {
      return t;
  }
private:
  //
  /*!
   * \brief System dimensions
   */
  int m, n;

  // Initial and current time
  double t0, t;

  // Discrete time step
  double dt;

  // Is the filter initialized?
  bool initialized;

  // n-size identity
  Eigen::MatrixXd I;

  // Estimated states
  Eigen::VectorXd x0, x1;
};
