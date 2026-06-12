/**
 * @file tags.cpp
 * @brief Implementation of MESHFEM3D element tagging system
 *
 * Implements parallel extraction of material classification from MESHFEM3D
 * Materials containers. Sets boundary tags to `none` since other boundary
 * conditions are not yet implemented.
 */
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

/**
 * @brief Constructor that extracts material and MPI tags
 *
 * Uses Kokkos parallel_for to extract material classification from the
 * Materials container. Sets all boundary tags to `none` since other
 * boundary conditions are not yet implemented. Marks elements appearing in
 * the adjacency graph's MPI connections as outer; all others remain inner.
 *
 * @param nspec Number of spectral elements
 * @param materials Materials container with material data
 * @param adjacency_graph Mesh-domain adjacency graph with MPI connections
 */
specfem::mesh::tags<specfem::element::dimension_tag::dim3>::tags(
    const int nspec, specfem::mesh::materials<dimension_tag> &materials,
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph) {

  this->nspec = nspec;

  this->tags_container =
      Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
          "specfem::mesh::tags::tags", this->nspec);

  Kokkos::parallel_for(
      "specfem::mesh::tags::copy_tags",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int ispec) {
        const auto [material_tag, property_tag, attenuation_tag] =
            materials.get_material_type(ispec);

        // Set to none since other boundary conditions are not yet implemented
        const auto boundary_tag = specfem::element::boundary_tag::none;

        // mpi_tag defaults to inner; outer elements are marked below
        this->tags_container(ispec) = { material_tag, property_tag,
                                        attenuation_tag, boundary_tag };
      });

  Kokkos::fence();

  // Mark elements touching an MPI partition boundary as outer. local_index is
  // in mesh domain, matching the indexing of tags_container.
  for (const auto &mpi_edge : adjacency_graph.mpi_connections()) {
    const int ispec_mesh = static_cast<int>(mpi_edge.local_index);
    this->tags_container(ispec_mesh).mpi_tag = specfem::element::mpi_tag::outer;
  }

  return;
}
