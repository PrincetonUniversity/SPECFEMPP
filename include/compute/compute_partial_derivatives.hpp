#ifndef _COMPUTE_PARTIAL_DERIVATIVES_HPP
#define _COMPUTE_PARTIAL_DERIVATIVES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "point/interface.hpp"
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
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z direction
  int ngllx; ///< Number of quadrature points in x direction
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
  partial_derivatives(const specfem::compute::mesh &mesh);
  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();

  template <bool load_jacobian>
  KOKKOS_INLINE_FUNCTION specfem::point::partial_derivatives2
  load_derivatives(const int ispec, const int iz, const int ix) const {
    if constexpr (load_jacobian) {
      return { xix(ispec, iz, ix), xiz(ispec, iz, ix), gammax(ispec, iz, ix),
               gammaz(ispec, iz, ix), jacobian(ispec, iz, ix) };
    } else {
      return { xix(ispec, iz, ix), xiz(ispec, iz, ix), gammax(ispec, iz, ix),
               gammaz(ispec, iz, ix) };
    }
  };
};
} // namespace compute
} // namespace specfem

#endif
