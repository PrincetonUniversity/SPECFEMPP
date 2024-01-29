#ifndef _COMPUTE_BOUNDARIES_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_BOUNDARIES_HPP

#include "compute/boundaries/impl/boundary_container.hpp"
#include "compute/boundaries/impl/boundary_container.tpp"
#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/specfem_enums.hpp"

namespace specfem {
namespace compute {
struct boundaries {

  boundaries() = default;

  boundaries(
      const int nspec, const specfem::compute::properties &properties,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  specfem::compute::impl::boundaries::boundary_container<
      specfem::enums::element::boundary_tag::acoustic_free_surface>
      acoustic_free_surface; ///< Acoustic free surface boundary

  specfem::compute::impl::boundaries::boundary_container<
      specfem::enums::element::boundary_tag::stacey>
      stacey; ///< Stacey boundary

  specfem::compute::impl::boundaries::boundary_container<
      specfem::enums::element::boundary_tag::composite_stacey_dirichlet>
      composite_stacey_dirichlet; ///< Composite Stacey and Dirichlet boundary
};

} // namespace compute
} // namespace specfem

#endif
