#pragma once

#include "adjacency_graph/adjacency_graph.hpp"
#include "boundaries/absorbing_boundary.hpp"
#include "globe.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"

#include <cstddef>
#include <set>
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
 * @brief A face identified by its zero-based mesh element and orientation.
 */
using face_key = std::pair<int, specfem::mesh_entity::dim3::type>;

/**
 * @brief Render a set of faces for an error message.
 *
 * @param faces Faces to list
 * @param limit Maximum number of faces to list before summarizing the rest
 * @return Space-separated list with one-based element numbers
 */
std::string describe_faces(const std::set<face_key> &faces,
                           const std::size_t limit);

/**
 * @brief Check that the chunk count is one the mesher can produce.
 *
 * SPECFEM3D_GLOBE meshes 1, 2, 3 or 6 chunks (@c read_compute_parameters).
 *
 * @param nchunks Chunk count recorded in the database
 * @throws std::runtime_error for any other value
 */
void check_chunk_count(const int nchunks);

/**
 * @brief Check that absorbing faces obey the mesher's Stacey rules.
 *
 * The mesher refuses absorbing conditions for the full Earth (6 chunks) and
 * does not support them for 3 chunks. For 1 or 2 chunks it places Stacey faces
 * on the crust/mantle and outer core only, never on the inner core.
 *
 * @param nchunks Chunk count recorded in the database
 * @param absorbing_boundary Absorbing faces of the mesh
 * @param element_context Per-element region context
 * @throws std::runtime_error if any rule is broken
 */
void check_absorbing_matches_chunk_count(
    const int nchunks,
    const specfem::mesh::absorbing_boundary<
        specfem::element::dimension_tag::dim3> &absorbing_boundary,
    const std::vector<specfem::mesh::globe_element_context> &element_context);

/**
 * @brief Check that every face of one surface also belongs to another.
 *
 * @param subset Surface whose faces must all be in @p superset
 * @param superset Surface that must contain every face of @p subset
 * @throws std::runtime_error listing the faces of @p subset missing from
 *         @p superset
 */
void check_surface_is_subset(const named_surface &subset,
                             const named_surface &superset);

/**
 * @brief Check that every face of a surface is a given face of an element in a
 * given region.
 *
 * @param surface Surface to check
 * @param region Region every owning element must belong to
 * @param face Face every entry must be
 * @param element_context Per-element region context
 * @throws std::runtime_error listing the entries that break either condition
 */
void check_surface_faces(
    const named_surface &surface, const specfem::element::region_tag region,
    const specfem::mesh_entity::dim3::type face,
    const std::vector<specfem::mesh::globe_element_context> &element_context);

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
