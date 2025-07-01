#pragma once

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::enums::element::boundary_tag boundary_tag>
specfem::assembly::impl::boundaries::boundary_container<boundary_tag>::
    boundary_container(
        const std::vector<specfem::enums::element::boundary_tag_container>
            &boundary_tags,
        const std::vector<specfem::point::boundary> &boundary_types) {

  const int nspec = boundary_tags.size();

  nelements = std::count(boundary_tags.begin(), boundary_tags.end(), value);

  if (nelements == 0) {
    return;
  }

  index_mapping = specfem::kokkos::DeviceView1d<int>(
      "specfem::assembly::boundaries::composite_stacey_dirichlet::ispec",
      nelements);
  boundary_type = specfem::kokkos::DeviceView1d<specfem::point::boundary>(
      "specfem::assembly::boundaries::composite_stacey_dirichlet::type",
      nelements);

  h_index_mapping = Kokkos::create_mirror_view(index_mapping);
  h_boundary_type = Kokkos::create_mirror_view(boundary_type);

  int index = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (boundary_tags[ispec] == value) {
      h_index_mapping(index) = ispec;
      h_boundary_type(index) = boundary_types[ispec];
      index++;
    }
  }

  assert(index == nelements);

  Kokkos::deep_copy(index_mapping, h_index_mapping);
  Kokkos::deep_copy(boundary_type, h_boundary_type);

  return;
}
