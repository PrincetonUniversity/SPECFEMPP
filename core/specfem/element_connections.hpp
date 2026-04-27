/**
 * @brief Connection types and mapping for spectral element mesh connectivity.
 *
 * Handles coordinate transformations between adjacent elements with different
 * conformity types (strongly/weakly conforming, nonconforming).
 */
#pragma once

#include "specfem/element.hpp"
#include "specfem/element_connections/tags.hpp"
#include "specfem/element_connections/to_string.hpp"
#include <string>

/**
 * @brief Element connectivity and coordinate mapping utilities.
 *
 * Provides connection types and coordinate transformations between adjacent
 * spectral elements in multi-physics computational domains. Handles conforming
 * and non-conforming element interfaces through compile-time mapping classes.
 */
namespace specfem::element_connections {

/**
 * @brief Convert connection type to string.
 * @param conn Connection type
 * @return String representation
 */
const std::string to_string(const specfem::element_connections::type &conn);

/**
 * @brief Coordinate mapping between adjacent spectral elements.
 *
 * Transforms coordinates between mesh entities (faces, edges) of adjacent
 * elements with different orientations.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 *
 * @code
 * // 2D example
 * connection_mapping<dim2> mapping(ngllz, ngllx, elem1_nodes, elem2_nodes);
 * auto [iz_mapped, ix_mapped] = mapping.map_coordinates(
 *     mesh_entity::left, mesh_entity::right, iz, ix);
 * @endcode
 */
template <specfem::element::dimension_tag DimensionTag>
class connection_mapping;

} // namespace specfem::element_connections

#include "element_connections/dim2/connections.hpp"
#include "element_connections/dim3/connections.hpp"

namespace specfem::element_connections {

/**
 * @brief Template argument deduction guides for automatic dimension detection.
 */
///@{
connection_mapping(const int, const int,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>)
    -> connection_mapping<specfem::element::dimension_tag::dim2>;

connection_mapping(const int, const int, const int,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>)
    -> connection_mapping<specfem::element::dimension_tag::dim3>;
///@}

} // namespace specfem::element_connections
