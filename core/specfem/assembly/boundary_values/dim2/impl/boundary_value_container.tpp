#pragma once

#include "boundary_value_container.hpp"
#include <Kokkos_Core.hpp>

template <specfem::element::boundary_tag BoundaryTag>
specfem::assembly::boundary_values_impl::boundary_value_container<
    specfem::element::dimension_tag::dim2, BoundaryTag>::
    boundary_value_container(
        const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types,
        const specfem::assembly::boundaries<dimension_tag> &boundaries)
    : property_index_mapping(
          "specfem::assembly::boundary_value_container::property_index_mapping",
          mesh.nspec),
      h_property_index_mapping(
          Kokkos::create_mirror_view(property_index_mapping)) {

  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    h_property_index_mapping(ispec) = -1;
  }

  specfem::tag_dispatch::for_each(
      combinations_by_medium, [&]<typename TagsType>() {
        container.template get<TagsType>() =
            BoundaryMediumTemplateType<TagsType>(nstep, mesh, element_types,
                                                 boundaries,
                                                 h_property_index_mapping);
      });

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
}
