#pragma once
#include "mesh_to_compute_mapping.hpp"
#include "specfem/element/tag_arrays.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include <vector>

specfem::assembly::mesh_impl::mesh_to_compute_mapping<
    specfem::element::dimension_tag::dim3>::
    mesh_to_compute_mapping(
        const specfem::mesh::tags<specfem::element::dimension_tag::dim3> &tags)
    : compute_to_mesh("specfem::assembly::compute_to_mesh_mapping", tags.nspec),
      mesh_to_compute("specfem::assembly::mesh_to_compute_mapping", tags.nspec),
      h_compute_to_mesh(Kokkos::create_mirror_view(compute_to_mesh)),
      h_mesh_to_compute(Kokkos::create_mirror_view(mesh_to_compute)) {

  const int nspec = tags.nspec;

  constexpr auto element_types =
      specfem::element::element_types<dimension_tag>();
  constexpr int total_element_types = element_types.size();

  std::array<std::vector<int>, total_element_types> element_type_ispec;
  int total_counted = 0;

  for (int i = 0; i < total_element_types; i++) {
    const auto [dimension_tag, medium_tag, property_tag, boundary_tag] =
        element_types[i];
    for (int ispec = 0; ispec < nspec; ispec++) {
      const auto tag = tags.tags_container(ispec);
      if (tag.medium_tag == medium_tag && tag.property_tag == property_tag &&
          tag.boundary_tag == boundary_tag) {
        element_type_ispec[i].push_back(ispec);
      }
    }
    total_counted += element_type_ispec[i].size();
  }

  assert(total_counted == nspec);

  int ispec = 0;

  for (const auto &element_ispec : element_type_ispec) {
    for (const auto &ispecs : element_ispec) {
      h_compute_to_mesh(ispec) = ispecs;
      h_mesh_to_compute(ispecs) = ispec;
      ispec++;
    }
  }

  Kokkos::deep_copy(compute_to_mesh, h_compute_to_mesh);
  Kokkos::deep_copy(mesh_to_compute, h_mesh_to_compute);

  assert(ispec == nspec);
}
