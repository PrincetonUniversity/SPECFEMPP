#pragma once

#include "boundary_value_container.hpp"
#include <Kokkos_Core.hpp>

template <specfem::element::boundary_tag BoundaryTag>
specfem::assembly::boundary_values_impl::boundary_value_container<
    specfem::dimension::type::dim2, BoundaryTag>::
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

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      CAPTURE(container) {
        _container_ = _boundary_medium_container<_dimension_tag_, _medium_tag_>(
            nstep, mesh, element_types, boundaries, h_property_index_mapping);
      });

  //   acoustic = specfem::assembly::impl::boundary_medium_container<
  //       DimensionTag, specfem::element::medium_tag::acoustic, BoundaryTag>(
  //       nstep, mesh, element_types, boundaries, h_property_index_mapping);

  //   elastic = specfem::assembly::impl::boundary_medium_container<
  //       DimensionTag, specfem::element::medium_tag::elastic_psv,
  //       BoundaryTag>( nstep, mesh, element_types, boundaries,
  //       h_property_index_mapping);

  //   poroelastic = specfem::assembly::impl::boundary_medium_container<
  //       DimensionTag, specfem::element::medium_tag::poroelastic,
  //       BoundaryTag>( nstep, mesh, element_types, boundaries,
  //       h_property_index_mapping);

  Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
}
