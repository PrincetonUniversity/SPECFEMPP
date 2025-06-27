#pragma once

#include "boundary_medium_container.hpp"

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::compute::impl::boundary_medium_container<DimensionTag, MediumTag,
                                            BoundaryTag>::
    boundary_medium_container(
        const int nstep, const specfem::compute::mesh mesh,
        const specfem::compute::element_types element_types,
        const specfem::compute::boundaries boundaries,
        specfem::kokkos::HostView1d<int> property_index_mapping) {

  int nelements = 0;
  const int nspec = mesh.nspec;
  const int nz = mesh.ngllz;
  const int nx = mesh.ngllx;

  for (int ispec = 0; ispec < nspec; ispec++) {
    if (element_types.get_medium_tag(ispec) == MediumTag &&
        element_types.get_boundary_tag(ispec) == BoundaryTag) {
      property_index_mapping(ispec) = nelements;
      nelements++;
    }
  }

  values = value_type("specfem::compute::boundary_medium_container::values",
                      nelements, nz, nx, nstep);

  h_values = Kokkos::create_mirror_view(values);

  return;
}
