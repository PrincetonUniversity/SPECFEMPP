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
 * @param boundaries Domain boundary face information (two sub-structs)
 */
specfem::mesh::tags<specfem::element::dimension_tag::dim3>::tags(
    const int nspec, specfem::mesh::materials<dimension_tag> &materials,
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
    const specfem::mesh::boundaries<dimension_tag> &boundaries) {

  this->nspec = nspec;

  this->tags_container =
      Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
          "specfem::mesh::tags::tags", this->nspec);

  std::vector<specfem::element::boundary_tag_container> bc(nspec);

  // All faces in the absorbing_boundary sub-struct → stacey
  const auto &abs = boundaries.absorbing_boundary;

  for (int i = 0; i < abs.nelements; ++i) {
    bc[abs.index_mapping(i)] += specfem::element::boundary_tag::stacey;
  }

  // Top-surface faces in acoustic_free_surface sub-struct → acoustic BC only
  // for acoustic elements (elastic elements at the top get natural Neumann BC)
  const auto &fs = boundaries.acoustic_free_surface;
  for (int i = 0; i < fs.nelem_acoustic_surface; ++i) {
    const int ispec = fs.index_mapping(i);
    const auto [medium_tag, property_tag, attenuation_tag] =
        materials.get_material_type(ispec);
    if (medium_tag == specfem::element::medium_tag::acoustic) {
      bc[ispec] += specfem::element::boundary_tag::acoustic_free_surface;
    }
  }

  for (int ispec = 0; ispec < nspec; ++ispec) {
    const auto [medium_tag, property_tag, attenuation_tag] =
        materials.get_material_type(ispec);
    this->tags_container(ispec) = { medium_tag, property_tag, attenuation_tag,
                                    bc[ispec].get_tag() };
  }

  // Mark elements touching an MPI partition boundary as outer. local_index is
  // in mesh domain, matching the indexing of tags_container.
  for (const auto &mpi_edge : adjacency_graph.mpi_connections()) {
    const int ispec_mesh = static_cast<int>(mpi_edge.local_index);
    this->tags_container(ispec_mesh).mpi_tag = specfem::element::mpi_tag::outer;
  }
}
