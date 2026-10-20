#pragma once

#include "adjacency_graph/adjacency_graph.hpp"
#include "globe.hpp"
#include "specfem/enums.hpp"

#include <string>
#include <utility>
#include <vector>

namespace specfem::mesh::globe_impl {

/**
 * @brief A database interface face list, together with the pair of regions it
 * separates.
 */
struct InterfaceSurface {
  /** @brief Name of the face list, used in error messages. */
  std::string name;

  /** @brief Face list read from the database. */
  const specfem::mesh::globe_boundary_surface *surface = nullptr;

  /** @brief Regions on the two sides of the interface, in either order. */
  std::pair<specfem::element::region_tag, specfem::element::region_tag>
      regions = { specfem::element::region_tag::crust_mantle,
                  specfem::element::region_tag::crust_mantle };
};

/**
 * @brief Check that each interface face list records exactly the faces where
 * the medium tag changes between its two regions.
 *
 * The database states where the medium changes twice: once implicitly, through
 * per-element medium tags and the adjacency graph, and once explicitly, as
 * interface face lists. After @c mesh_dim3_base::setup_coupled_interfaces has
 * promoted every edge between elements of different medium tags to
 * @c element_connections::type::weakly_conforming, the implied faces are every
 * such edge's (source element, @c orientation) pair. Each implied face is
 * attributed to the surface whose region pair matches the regions of its two
 * elements, so a face filed under the wrong surface is rejected. Only faces are
 * compared; fluid and solid elements also meet at edges and corners, but the
 * database does not record those.
 *
 * Each surface list must carry both sides of its interface, so that it is
 * comparable with the implied faces, which see every face from both of its
 * elements.
 *
 * @param adjacency_graph Adjacency graph after coupled-interface setup
 * @param element_context Per-element region context
 * @param surfaces Database face lists that together cover every interface
 * @throws std::runtime_error naming the counts and the first few differing
 *         faces, with one-based database element numbers
 */
void check_interface_surfaces_match_medium_tags(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_element_context> &element_context,
    const std::vector<InterfaceSurface> &surfaces);

/**
 * @brief Reject meshes that split a fluid-solid interface across MPI ranks.
 *
 * Coupling is assembled from local connections only, so an interface face
 * shared between two ranks would silently contribute no coupling term. The
 * neighbor's medium is unknown on this rank, but in the globe every
 * fluid-solid face has an outer-core element on one side and is radial (top or
 * bottom), so an outer-core element with a radial face on an MPI boundary is
 * enough to flag it.
 *
 * @param adjacency_graph Adjacency graph carrying the MPI connections
 * @param element_context Per-element region context
 * @throws std::runtime_error on the rank owning the fluid side
 */
void check_no_cross_rank_fluid_solid_faces(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<specfem::mesh::globe_element_context> &element_context);

} // namespace specfem::mesh::globe_impl
