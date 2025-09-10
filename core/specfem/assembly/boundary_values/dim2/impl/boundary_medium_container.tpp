#pragma once

#include "boundary_medium_container.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"

template <specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::boundary_values_impl::boundary_medium_container<
    specfem::dimension::type::dim2, MediumTag, BoundaryTag>::
    boundary_medium_container(
        const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::assembly::element_types<dimension_tag> &element_types,
        const specfem::assembly::boundaries<dimension_tag> boundaries,
        Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
            property_index_mapping) {

  int nelements = 0;
  const int nspec = mesh.nspec;
  const int nz = mesh.element_grid.ngllz;
  const int nx = mesh.element_grid.ngllx;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types.get_medium_tag(ispec) == MediumTag &&
        element_types.get_boundary_tag(ispec) == BoundaryTag) {
      property_index_mapping(ispec) = nelements;
      nelements++;
    }
  }

  values = ValueViewType("specfem::assembly::boundary_medium_container::values",
                      nelements, nz, nx, nstep);

  h_values = Kokkos::create_mirror_view(values);

  return;
}
