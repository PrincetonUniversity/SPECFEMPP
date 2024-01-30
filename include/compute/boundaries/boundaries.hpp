#ifndef _COMPUTE_BOUNDARIES_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_BOUNDARIES_HPP

#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"
#include "point/boundary.hpp"

namespace specfem {
namespace compute {
struct boundaries {

  boundaries() = default;

  boundaries(
      const int nspec, const specfem::compute::properties &properties,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  specfem::kokkos::DeviceView1d<specfem::enums::element::boundary_tag_container>
      boundary_tags; ///< Boundary tags for each element
  specfem::kokkos::HostMirror1d<specfem::enums::element::boundary_tag_container>
      h_boundary_tags; ///< Host mirror of boundary tags

  specfem::kokkos::DeviceView1d<specfem::point::boundary>
      boundary_types; ///< Boundary types for each element
  specfem::kokkos::HostMirror1d<specfem::point::boundary>
      h_boundary_types; ///< Host mirror of boundary types

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::acoustic_free_surface>
  //       acoustic_free_surface; ///< Acoustic free surface boundary

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::stacey>
  //       stacey; ///< Stacey boundary

  //   specfem::compute::impl::boundaries::boundary_container<
  //       specfem::enums::element::boundary_tag::composite_stacey_dirichlet>
  //       composite_stacey_dirichlet; ///< Composite Stacey and Dirichlet
  //       boundary
};

} // namespace compute
} // namespace specfem

#endif
