#pragma once
#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "enumerations/medium_tags.hpp"
#include "mesh_to_compute_mapping.hpp"
#include <vector>

specfem::assembly::mesh_impl::mesh_to_compute_mapping<
    specfem::dimension::type::dim2>::
    mesh_to_compute_mapping(
        const specfem::mesh::tags<specfem::dimension::type::dim2> &tags)
    : compute_to_mesh("specfem::assembly::mesh_to_compute_mapping", tags.nspec),
      mesh_to_compute("specfem::assembly::mesh_to_compute_mapping",
                      tags.nspec) {

  const int nspec = tags.nspec;

  constexpr auto element_types = specfem::element::element_types<dimension_tag>();
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
      compute_to_mesh(ispec) = ispecs;
      mesh_to_compute(ispecs) = ispec;
      ispec++;
    }
  }

  assert(ispec == nspec);
}
