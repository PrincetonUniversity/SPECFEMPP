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
  virtual Kokkos::View<type_real *, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace>
  get_xi() const {
    return this->xi;
  };
  /**
   * Get quadrature weights on device
   *
   */
  virtual Kokkos::View<type_real *, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace>
  get_w() const {
    return this->w;
  };
  /**
   * Get derivatives of quadrature polynomials at quadrature points on device
   *
   */
  virtual Kokkos::View<type_real **, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace>
  get_hprime() const {
    return this->hprime;
  };
  /**
   * Get quadrature points on host
   *
   */
  virtual Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_hxi() const {
    return this->h_xi;
  };
  /**
   * Get quadrature weights on host
   *
   */
  virtual Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_hw() const {
    return this->h_w;
  };
  /**
   * Get derivatives of quadrature polynomials at quadrature points on host
   *
   */
  virtual Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_hhprime() const {
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

  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      xi; ///< qudrature points stored on
          ///< device
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
      h_xi; ///< quadrature points stored
            ///< on host

  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      w; ///< qudrature weights stored on
         ///< device
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
      h_w; ///< quadrature weights stored on
           ///< host

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>
      hprime; ///< Polynomial derivatives
              ///< stored on device
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      h_hprime; ///< Polynomial derivatives
                ///< store on host
};

std::ostream &operator<<(std::ostream &out,
                         specfem::quadrature::quadrature &quad);
} // namespace quadrature
} // namespace specfem
