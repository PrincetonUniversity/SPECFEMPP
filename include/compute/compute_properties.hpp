#ifndef _COMPUTE_PROPERTIES_HPP
#define _COMPUTE_PROPERTIES_HPP

#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem {
namespace compute {

/**
 * @brief Material properties stored at every quadrature point
 *
 */
struct properties {
  /**
   * @name Material properties
   *
   * h_ prefixes denote views stored on host
   */
  ///@{
  specfem::kokkos::DeviceView3d<type_real> rho;
  specfem::kokkos::HostMirror3d<type_real> h_rho;

  specfem::kokkos::DeviceView3d<type_real> mu;
  specfem::kokkos::HostMirror3d<type_real> h_mu;

  specfem::kokkos::HostView3d<type_real> kappa;

  specfem::kokkos::HostView3d<type_real> qmu;

  specfem::kokkos::HostView3d<type_real> qkappa;

  specfem::kokkos::HostView3d<type_real> rho_vp;

  specfem::kokkos::HostView3d<type_real> rho_vs;

  specfem::kokkos::DeviceView3d<type_real> lambdaplus2mu;
  specfem::kokkos::HostMirror3d<type_real> h_lambdaplus2mu;
  ///@}
  // element type is defined in specfem_setup.hpp
  specfem::kokkos::DeviceView1d<specfem::enums::element::type>
      ispec_type; ///< type of element
                  ///< stored on device
  specfem::kokkos::HostMirror1d<specfem::enums::element::type>
      h_ispec_type; ///< type of element
                    ///< stored on host

  /**
   * @brief Default constructor
   *
   */
  properties(){};
  /**
   * @brief Constructor to allocate views
   *
   * @param nspec Number for spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   */
  properties(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Constructor to allocate and assign views
   *
   * @param kmato Material specification number
   * @param materials Pointer to material objects read from database file
   * @param nspec Number for spectral elements
   * @param ngllz Number of quadrature points in z dimension
   * @param ngllx Number of quadrature points in x dimension
   */
  properties(const specfem::kokkos::HostView1d<int> kmato,
             const std::vector<specfem::material::material *> &materials,
             const int nspec, const int ngllx, const int ngllz);

  /**
   * @brief Helper routine to sync views within this struct
   *
   */
  void sync_views();
};

} // namespace compute
} // namespace specfem

#endif
