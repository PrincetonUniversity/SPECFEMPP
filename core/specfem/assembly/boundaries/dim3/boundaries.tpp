#pragma once

#include "boundaries.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::boundaries<specfem::dimension::type::dim3>::boundaries(
    const int nspec, const int ngllz, const int nglly, const int ngllx,
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
    const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix)
    : boundary_tags("specfem::assembly::boundaries::boundary_tags", nspec),
      h_boundary_tags(Kokkos::create_mirror_view(boundary_tags)) {

  std::vector<specfem::element::boundary_tag_container> boundary_tag(nspec);

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->h_boundary_tags(ispec) = boundary_tag[ispec].get_tag();
  }

  // Check if mesh and compute boundary tags match
  for (int ispec = 0; ispec < nspec; ++ispec) {
    // In dim3, mesh and compute indices are the same (no reordering)
    const int ispec_compute = ispec;
    const auto m_boundary_tag = mesh.tags.tags_container(ispec).boundary_tag;
    const auto c_boundary_tag = this->h_boundary_tags(ispec_compute);
    if (m_boundary_tag != c_boundary_tag) {
      std::cout << "ispec: " << ispec << std::endl;
      std::cout << "m_boundary_tag: " << specfem::element::to_string(m_boundary_tag) << std::endl;
      std::cout << "c_boundary_tag: " << specfem::element::to_string(c_boundary_tag) << std::endl;
      throw std::runtime_error("Mesh and compute boundary tags do not match");
    }
  }

  Kokkos::deep_copy(this->boundary_tags, this->h_boundary_tags);
  return;
}
