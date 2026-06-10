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
        const specfem::mesh::tags<specfem::element::dimension_tag::dim3> &tags,
        const specfem::mesh::adjacency_graph<
            specfem::element::dimension_tag::dim3> &adjacency_graph)
    : compute_to_mesh("specfem::assembly::compute_to_mesh_mapping", tags.nspec),
      mesh_to_compute("specfem::assembly::mesh_to_compute_mapping", tags.nspec),
      h_compute_to_mesh(Kokkos::create_mirror_view(compute_to_mesh)),
      h_mesh_to_compute(Kokkos::create_mirror_view(mesh_to_compute)) {

  const int nspec = tags.nspec;

  // Identify outer (MPI-boundary) elements: any element appearing as a local
  // element in an MPI connection. This must match the classification used in
  // specfem::assembly::element_types (which also uses mpi_connections()).
  std::vector<bool> is_outer(nspec, false);
  for (const auto &mpi_edge : adjacency_graph.mpi_connections()) {
    is_outer[static_cast<int>(mpi_edge.local_index)] = true;
  }

  // Here we need to put ALL element combinations!
  using ET = decltype(
      DIMENSION_SET(dim3) *
      MEDIUM_SET(elastic, acoustic, elastic_spin) *
      PROPERTY_SET(isotropic, isotropic_cosserat) *
      BOUNDARY_SET(none));
  constexpr auto element_types = ET::combos;
  constexpr int total_element_types = ET::size;

  std::array<std::vector<int>, total_element_types> element_type_ispec;
  int total_counted = 0;

  for (int i = 0; i < total_element_types; i++) {
    const auto medium_tag   = element_types[i].template get<1>();
    const auto property_tag = element_types[i].template get<2>();
    const auto boundary_tag = element_types[i].template get<3>();
    // Emit inner (partition-interior) elements first, then outer (MPI-boundary)
    // elements, so that each (medium, property, boundary, mpi_tag) sublist is a
    // contiguous element range. The stiffness kernels iterate these sublists
    // with SIMD, where lane L is mesh element `ispec + L`; a non-contiguous
    // list would make the lanes read the wrong elements.
    for (int pass = 0; pass < 2; pass++) {
      const bool want_outer = (pass == 1);
      for (int ispec = 0; ispec < nspec; ispec++) {
        const auto tag = tags.tags_container(ispec);
        if (tag.medium_tag == medium_tag && tag.property_tag == property_tag &&
            tag.boundary_tag == boundary_tag && is_outer[ispec] == want_outer) {
          element_type_ispec[i].push_back(ispec);
        }
      }
    }
    total_counted += element_type_ispec[i].size();
  }

  if (total_counted != nspec) {
    const std::string msg =
        "specfem::assembly::mesh_to_compute_mapping: only " +
        std::to_string(total_counted) + " of " + std::to_string(nspec) +
        " elements matched a known (medium, property, boundary) combination. "
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
