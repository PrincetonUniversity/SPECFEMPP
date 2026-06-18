#pragma once
#include "mesh_to_compute_mapping.hpp"
#include "specfem/enums.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/mesh.hpp"
#include "specfem/program/abort.hpp"
#include "specfem/tag_dispatch.hpp"
#include <string>
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

  // Here we need to put ALL element combinations!
  using ET = decltype(
      DIMENSION_SET(dim3) *
      MEDIUM_SET(elastic, acoustic, elastic_spin) *
      PROPERTY_SET(isotropic, isotropic_cosserat) *
      ATTENUATION_SET(none, constant_isotropic) *
      BOUNDARY_SET(none) *
      MPI_SET(inner, outer));
  constexpr auto element_types = ET::combos;
  constexpr int total_element_types = ET::size;

  std::array<std::vector<int>, total_element_types> element_type_ispec;
  int total_counted = 0;

  for (int i = 0; i < total_element_types; i++) {
    const auto medium_tag      = element_types[i].template get<1>();
    const auto property_tag    = element_types[i].template get<2>();
    const auto attenuation_tag = element_types[i].template get<3>();
    const auto boundary_tag    = element_types[i].template get<4>();
    const auto mpi_tag         = element_types[i].template get<5>();
    for (int ispec = 0; ispec < nspec; ispec++) {
      const auto tag = tags.tags_container(ispec);
      if (tag.medium_tag == medium_tag && tag.property_tag == property_tag &&
          tag.attenuation_tag == attenuation_tag &&
          tag.boundary_tag == boundary_tag && tag.mpi_tag == mpi_tag) {
        element_type_ispec[i].push_back(ispec);
      }
    }
    total_counted += element_type_ispec[i].size();
  }

  if (total_counted != nspec) {
    const std::string msg =
        "specfem::assembly::mesh_to_compute_mapping: only " +
        std::to_string(total_counted) + " of " + std::to_string(nspec) +
        " elements matched a known (medium, property, attenuation, boundary, "
        "mpi) combination. "
        "The compute<->mesh index mapping would be left partially "
        "uninitialized. This usually means an element carries a tag "
        "combination not present in the supported element-type set.";
    specfem::Logger::error(msg);
    specfem::program::abort(msg);
  }

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

  if (ispec != nspec) {
    const std::string msg =
        "specfem::assembly::mesh_to_compute_mapping: assigned " +
        std::to_string(ispec) + " compute indices but expected " +
        std::to_string(nspec) + ". The compute<->mesh index mapping is not a "
        "valid bijection.";
    specfem::Logger::error(msg);
    specfem::program::abort(msg);
  }
}
