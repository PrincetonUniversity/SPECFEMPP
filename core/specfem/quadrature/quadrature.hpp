#pragma once

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace quadrature {

/**
 * @brief Base quadrature class
 *
 */
class quadrature {
public:
  /**
   * @brief Construct a quadrature object with default values
   *
   */
  quadrature() {};
  /**
   * @brief Construct a new quadrature object
   *
   * @param alpha alpha value of the quadrature
   * @param beta beta value of the quadrature
   */
  virtual specfem::kokkos::DeviceView1d<type_real> get_xi() const {
    return this->xi;
  };
  /**
   * Get quadrature weights on device
   *
   */
  virtual specfem::kokkos::DeviceView1d<type_real> get_w() const {
    return this->w;
  };
  /**
   * Get derivatives of quadrature polynomials at quadrature points on device
   *
   */
  virtual specfem::kokkos::DeviceView2d<type_real> get_hprime() const {
    return this->hprime;
  };
  /**
   * Get quadrature points on host
   *
   */
  virtual specfem::kokkos::HostMirror1d<type_real> get_hxi() const {
    return this->h_xi;
  };
  /**
   * Get quadrature weights on host
   *
   */
  virtual specfem::kokkos::HostMirror1d<type_real> get_hw() const {
    return this->h_w;
  };
  /**
   * Get derivatives of quadrature polynomials at quadrature points on host
   *
   */
  virtual specfem::kokkos::HostMirror2d<type_real> get_hhprime() const {
    return this->h_hprime;
  };
  /**
   * @brief get number of quadrture points
   *
   */
  virtual int get_N() const { return this->N; };
  /**
   * @brief Log quadrature information to console
   */
  virtual void print(std::ostream &out) const;

  /**
   * @brief return string representation of the quadrature
   * @return std::string String representation of the quadrature
   */
  virtual std::string to_string() const;

  // typedef polynomial = specfem::quadrature::polynomial::Lagrange;

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
};

std::ostream &operator<<(std::ostream &out,
                         specfem::quadrature::quadrature &quad);
} // namespace quadrature
} // namespace specfem
