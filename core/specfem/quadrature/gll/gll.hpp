#pragma once

#include "../quadrature.hpp"
#include "kokkos_abstractions.h"
#include "lagrange_poly.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace quadrature {
namespace gll {

class gll : public quadrature {
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
  gll();
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value of the quadrature
   * @param beta beta value of the quadrature
   */
  gll(const type_real alpha, const type_real beta);
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value of the quadrature
   * @param beta beta value of quadrature
   * @param N Number of quadrature points
   */
  gll(const type_real alpha, const type_real beta, const int N);
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
  specfem::kokkos::DeviceView1d<type_real> get_xi() const override {
    return this->xi;
  }
  /**
   * Get quadrature weights on device
   *
   */
  specfem::kokkos::DeviceView1d<type_real> get_w() const override {
    return this->w;
  }
  /**
   * Get derivatives of quadrature polynomials at quadrature points on device
   *
   */
  specfem::kokkos::DeviceView2d<type_real> get_hprime() const override {
    return this->hprime;
  }
  /**
   * Get quadrature points on host
   *
   */
  specfem::kokkos::HostMirror1d<type_real> get_hxi() const override {
    return this->h_xi;
  }
  /**
   * Get quadrature weights on host
   *
   */
  specfem::kokkos::HostMirror1d<type_real> get_hw() const override {
    return this->h_w;
  }
  /**
   * Get derivatives of quadrature polynomials at quadrature points on host
   *
   */
  specfem::kokkos::HostMirror2d<type_real> get_hhprime() const override {
    return this->h_hprime;
  }
  /**
   * @brief get number of quadrture points
   *
   */
  int get_N() const override { return this->N; }
  /**
   * @brief Log GLL quadrature information to console
   */
  void print(std::ostream &out) const override;

  /**
   * @brief return string representation of the GLL quadrature
   * @return std::string String representation of the GLL quadrature
   */
  std::string to_string() const override;

private:
  type_real alpha; ///< alpha value of the quadrature
  type_real beta;  ///< beta value of the quadrature
  int N;           ///< Number of qudrature points

  specfem::kokkos::DeviceView1d<type_real> xi;   ///< qudrature points stored on
                                                 ///< device
  specfem::kokkos::HostMirror1d<type_real> h_xi; ///< quadrature points stored
                                                 ///< on host

  specfem::kokkos::DeviceView1d<type_real> w; ///< qudrature weights stored on
                                              ///< device
  specfem::kokkos::HostView1d<type_real> h_w; ///< quadrature weights stored on
                                              ///< host

  specfem::kokkos::DeviceView2d<type_real> hprime; ///< Polynomial derivatives
                                                   ///< stored on device
  specfem::kokkos::HostView2d<type_real> h_hprime; ///< Polynomial derivatives
                                                   ///< store on host

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
} // namespace gll
} // namespace quadrature
} // namespace specfem
