#ifndef _COMPUTE_PARTIAL_DERIVATIVES_HPP
#define _COMPUTE_PARTIAL_DERIVATIVES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "quadrature/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {
/**
 * @brief Partial derivates matrices required to compute integrals
 *
 * The matrices are stored in (ispec, iz, ix) format
 *
 */
struct partial_derivatives {
  specfem::kokkos::DeviceView3d<type_real> xix; ///< inverted partial derivates
                                                ///< @xix stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xix; ///< inverted partial
                                                  ///< derivates @xix stored on
                                                  ///< the host
  specfem::kokkos::DeviceView3d<type_real> xiz; ///< inverted partial derivates
                                                ///< @xiz stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xiz;  ///< inverted partial
                                                   ///< derivates @xiz stored on
                                                   ///< the host
  specfem::kokkos::DeviceView3d<type_real> gammax; ///< inverted partial
                                                   ///< derivates @gammax
                                                   ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammax; ///< inverted partial
                                                     ///< derivates @gammax
                                                     ///< stored on host
  specfem::kokkos::DeviceView3d<type_real> gammaz;   ///< inverted partial
                                                     ///< derivates @gammaz
                                                     ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammaz; ///< inverted partial
                                                     ///< derivates @gammaz
                                                     ///< stored on host
  specfem::kokkos::DeviceView3d<type_real> jacobian; ///< Jacobian values stored
                                                     ///< on device
  specfem::kokkos::HostMirror3d<type_real> h_jacobian; ///< Jacobian values
                                                       ///< stored on host
  /**
   * @brief Default constructor
   *
   */
  partial_derivatives(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number of spectral elements
   * @param ngllz Number of quadrature points in z direction
   * @param ngllx Number of quadrature points in x direction
   */
  partial_derivatives(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param coorg (x,z) for every spectral element control node
   * @param knods Global control element number for every control node
   * @param quadx Quadrature object in x dimension
   * @param quadz Quadrature object in z dimension
   */
  partial_derivatives(const specfem::compute::mesh &mesh,
                      const specfem::compute::quadrature &quadrature);
  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};

/**
 * @brief Struct to store the partial derivatives of the element at a given
 * quadrature point
 *
 */
struct element_partial_derivatives {
  type_real xix;
  type_real gammax;
  type_real xiz;
  type_real gammaz;
  type_real jacobian;

  KOKKOS_FUNCTION
  element_partial_derivatives() = default;

  KOKKOS_FUNCTION
  element_partial_derivatives(const type_real &xix, const type_real &gammax,
                              const type_real &xiz, const type_real &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  KOKKOS_FUNCTION
  element_partial_derivatives(const type_real &xix, const type_real &gammax,
                              const type_real &xiz, const type_real &gammaz,
                              const type_real &jacobian)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz), jacobian(jacobian) {
  }

  // KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
  // specfem::compute::element_partial_derivatives::compute_normal(
  //     const specfem::enums::boundaries::type type) const;

  template <specfem::enums::boundaries::type type>
  KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
  compute_normal() const {
    ASSERT(false, "Invalid boundary type");
    return specfem::kokkos::array_type<type_real, 2>();
  };
};
} // namespace compute
} // namespace specfem

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::compute::element_partial_derivatives::compute_normal<
    specfem::enums::boundaries::type::BOTTOM>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = -1.0 * this->gammax * this->jacobian;
  dn[1] = -1.0 * this->gammaz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::compute::element_partial_derivatives::compute_normal<
    specfem::enums::boundaries::type::TOP>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = this->gammax * this->jacobian;
  dn[1] = this->gammaz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::compute::element_partial_derivatives::compute_normal<
    specfem::enums::boundaries::type::LEFT>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = -1.0 * this->xix * this->jacobian;
  dn[1] = -1.0 * this->xiz * this->jacobian;
  return dn;
};

template <>
KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::compute::element_partial_derivatives::compute_normal<
    specfem::enums::boundaries::type::RIGHT>() const {
  specfem::kokkos::array_type<type_real, 2> dn;
  dn[0] = this->xix * this->jacobian;
  dn[1] = this->xiz * this->jacobian;
  return dn;
};

#endif
