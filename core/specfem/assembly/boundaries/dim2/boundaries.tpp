#pragma once

#include "boundaries.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::boundaries<specfem::dimension::type::dim2>::boundaries(
    const int nspec, const int ngllz, const int ngllx,
    const specfem::mesh::mesh<dimension_tag> &mesh,
    const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
    const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix)
    : boundary_tags("specfem::assembly::boundaries::boundary_tags", nspec),
      h_boundary_tags(Kokkos::create_mirror_view(boundary_tags)),
      acoustic_free_surface_index_mapping(
          "specfem::assembly::boundaries::acoustic_free_surface_index_mapping",
          nspec),
      h_acoustic_free_surface_index_mapping(
          Kokkos::create_mirror_view(acoustic_free_surface_index_mapping)),
      stacey_index_mapping(
          "specfem::assembly::boundaries::stacey_index_mapping", nspec),
      h_stacey_index_mapping(Kokkos::create_mirror_view(stacey_index_mapping)) {

  std::vector<specfem::element::boundary_tag_container> boundary_tag(nspec);

  this->acoustic_free_surface =
      specfem::assembly::boundaries_impl::acoustic_free_surface<dimension_tag>(
          nspec, ngllz, ngllx, mesh.boundaries.acoustic_free_surface,
          mesh_assembly, this->h_acoustic_free_surface_index_mapping,
          boundary_tag);

  this->stacey = specfem::assembly::boundaries_impl::stacey<dimension_tag>(
      nspec, ngllz, ngllx, mesh.boundaries.absorbing_boundary, mesh_assembly,
      jacobian_matrix, this->h_stacey_index_mapping, boundary_tag);

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->h_boundary_tags(ispec) = boundary_tag[ispec].get_tag();
  }

  // Check if mesh and compute boundary tags match
  for (int ispec = 0; ispec < nspec; ++ispec) {
    const int ispec_compute = mesh_assembly.mesh_to_compute(ispec);
    const auto m_boundary_tag = mesh.tags.tags_container(ispec).boundary_tag;
    const auto c_boundary_tag = this->h_boundary_tags(ispec_compute);
    if (m_boundary_tag != c_boundary_tag) {
      throw std::runtime_error("Mesh and compute boundary tags do not match");
    }
  }

  Kokkos::deep_copy(this->acoustic_free_surface_index_mapping,
                    this->h_acoustic_free_surface_index_mapping);

  Kokkos::deep_copy(this->stacey_index_mapping, this->h_stacey_index_mapping);
  Kokkos::deep_copy(this->boundary_tags, this->h_boundary_tags);
  return;
}
