#ifndef _COMPUTE_PARTIAL_DERIVATIVES_HPP
#define _COMPUTE_PARTIAL_DERIVATIVES_HPP

#include "kokkos_abstractions.h"
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
                                                ///< \f$\partial \xi / \partial
                                                ///< x\f$ stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xix; ///< inverted partial
                                                  ///< derivates \f$\partial \xi
                                                  ///< / \partial x\f$ stored on
                                                  ///< the host
  specfem::kokkos::DeviceView3d<type_real> xiz; ///< inverted partial derivates
                                                ///< \f$\partial \xi / \partial
                                                ///< z\f$ stored on the device
  specfem::kokkos::HostMirror3d<type_real> h_xiz; ///< inverted partial
                                                  ///< derivates \f$\partial \xi
                                                  ///< / \partial z\f$ stored on
                                                  ///< the host
  specfem::kokkos::DeviceView3d<type_real> gammax;   ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial x\f$
                                                     ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammax; ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial x\f$
                                                     ///< stored on host
  specfem::kokkos::DeviceView3d<type_real> gammaz;   ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial z\f$
                                                     ///< stored on device
  specfem::kokkos::HostMirror3d<type_real> h_gammaz; ///< inverted partial
                                                     ///< derivates \f$\partial
                                                     ///< \gamma / \partial z\f$
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
  partial_derivatives(const specfem::kokkos::HostView2d<type_real> coorg,
                      const specfem::kokkos::HostView2d<int> knods,
                      const specfem::quadrature::quadrature *quadx,
                      const specfem::quadrature::quadrature *quadz);

  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};
} // namespace compute
} // namespace specfem

#endif
