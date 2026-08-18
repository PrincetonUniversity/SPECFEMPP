#pragma once

#include "boundaries.hpp"
#include "impl/acoustic_free_surface.hpp"
#include "impl/stacey.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::assembly::boundaries<specfem::element::dimension_tag::dim3>::boundaries(
    const int nspec, const int ngllz, const int nglly, const int ngllx,
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
      h_stacey_index_mapping(
          Kokkos::create_mirror_view(stacey_index_mapping)) {

  std::vector<specfem::element::boundary_tag_container> boundary_tag(nspec);

  this->acoustic_free_surface =
      specfem::assembly::boundaries_impl::acoustic_free_surface<dimension_tag>(
          nspec, ngllz, nglly, ngllx, mesh, mesh_assembly,
          this->h_acoustic_free_surface_index_mapping, boundary_tag);

  this->stacey = specfem::assembly::boundaries_impl::stacey<dimension_tag>(
      nspec, ngllz, nglly, ngllx, mesh, mesh_assembly, jacobian_matrix,
      this->h_stacey_index_mapping, boundary_tag);

  for (int ispec = 0; ispec < nspec; ispec++) {
    this->h_boundary_tags(ispec) = boundary_tag[ispec].get_tag();
  }

  Kokkos::deep_copy(this->acoustic_free_surface_index_mapping,
                    this->h_acoustic_free_surface_index_mapping);
  Kokkos::deep_copy(this->stacey_index_mapping, this->h_stacey_index_mapping);
  Kokkos::deep_copy(this->boundary_tags, this->h_boundary_tags);
  return;
}
