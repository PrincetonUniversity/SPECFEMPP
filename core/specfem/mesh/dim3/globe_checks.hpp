#pragma once

#include "adjacency_graph/adjacency_graph.hpp"
#include "globe.hpp"
#include "specfem/enums.hpp"

#include <string>
#include <utility>
#include <vector>

namespace specfem::mesh::globe_impl {

/**
 * @brief A named face list from the database, used in error messages.
 */
using named_surface =
    std::pair<std::string, const specfem::mesh::globe_boundary_surface *>;

/**
 * @brief Check that the interface faces implied by medium contrast are exactly
 * the faces recorded in the given surface lists.
 *
 * The database states where the medium changes twice: once implicitly, through
 * per-element medium tags and the adjacency graph, and once explicitly, as
 * interface face lists. After @c mesh_dim3_base::setup_coupled_interfaces has
 * promoted every medium-contrast edge to
 * @c element_connections::type::weakly_conforming, the implied set is every
 * such edge's (source element, @c orientation) pair. Only faces are compared;
 * fluid and solid elements also meet at edges and corners, but the database
 * does not record those.
 *
 * The surface lists must carry both sides of each interface, so that their
 * union is comparable with the implied set, which sees every face from both of
 * its elements.
 *
 * @param adjacency_graph Adjacency graph after coupled-interface setup
 * @param surfaces Database face lists that together cover every interface
 * @throws std::runtime_error naming the counts and the first few differing
 *         faces, with one-based database element numbers
 */
void check_interfaces_match_medium_contrast(
    const specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
        &adjacency_graph,
    const std::vector<named_surface> &surfaces);

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
