#ifndef _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_CONTAINER_TPP
#define _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_CONTAINER_TPP

#include "boundary_values_container.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType,
          specfem::element::boundary_tag BoundaryTag>
specfem::compute::boundary_value_container<DimensionType, BoundaryTag>::
    boundary_value_container(const int nstep, const specfem::compute::mesh mesh,
                             const specfem::compute::element_types element_types,
                             const specfem::compute::boundaries boundaries)
    : property_index_mapping(
          "specfem::compute::boundary_value_container::property_index_mapping",
          mesh.nspec),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {

  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  Kokkos::fence();

  acoustic = specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::acoustic, BoundaryTag>(
      nstep, mesh, element_types, boundaries, h_property_index_mapping);

  elastic = specfem::compute::impl::boundary_medium_container<
      DimensionType, specfem::element::medium_tag::elastic_sv, BoundaryTag>(
      nstep, mesh, element_types, boundaries, h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
}

#endif
