#ifndef QUADRATURE_H
#define QUADRATURE_H

#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace quadrature {

class quadrature {
  /**
   * @brief Defines the GLL/GLJ quadrature and related matrices required for
   * quadrature integration
   *
   */
public:
  /**
   * @brief Construct a quadrature object with default values
   *
   * Default values: alpha = 0, beta = 0, N = 5
   *
   */
  quadrature();
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value of the quadrature
   * @param beta beta value of the quadrature
   */
  quadrature(const type_real alpha, const type_real beta);
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value of the quadrature
   * @param beta beta value of quadrature
   * @param N Number of quadrature points
   */
  quadrature(const type_real alpha, const type_real beta, const int N);
  /**
   * @brief Set the derivation matrices
   *
   * Set the matrices required for compute integrals
   *
   */
  void set_derivation_matrices();
  /**
   * Get quadrature points on device
   *
   */
  specfem::DeviceView1d<type_real> get_xi() const;
  /**
   * Get quadrature weights on device
   *
   */
  specfem::DeviceView1d<type_real> get_w() const;
  /**
   * Get derivatives of quadrature polynomials at quadrature points on device
   *
   */
  specfem::DeviceView2d<type_real> get_hprime() const;
  /**
   * Get quadrature points on host
   *
   */
  specfem::HostMirror1d<type_real> get_hxi() const;
  /**
   * Get quadrature weights on host
   *
   */
  specfem::HostMirror1d<type_real> get_hw() const;
  /**
   * Get derivatives of quadrature polynomials at quadrature points on host
   *
   */
  specfem::HostMirror2d<type_real> get_hhprime() const;
  /**
   * @brief get number of quadrture points
   *
   */
  int get_N() const;

private:
  type_real alpha; ///< alpha value of the quadrature
  type_real beta;  ///< beta value of the quadrature
  int N;           ///< Number of qudrature points

  specfem::DeviceView1d<type_real> xi;   ///< qudrature points stored on device
  specfem::HostMirror1d<type_real> h_xi; ///< quadrature points stored on host

  specfem::DeviceView1d<type_real> w;   ///< qudrature weights stored on device
  specfem::HostMirror1d<type_real> h_w; ///< quadrature weights stored on host

  specfem::DeviceView2d<type_real> hprime; ///< Polynomial derivatives stored on
                                           ///< device
  specfem::HostMirror2d<type_real> h_hprime; ///< Polynomial derivatives store
                                             ///< on host

  /**
   * Set View allocations for all derivative matrices
   *
   */
  void set_allocations();
  /**
   * Sync views between device and host
   *
   */
  void sync_views();
};
} // namespace quadrature

#endif
