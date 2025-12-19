/**
 * @brief Connection types and mapping for spectral element mesh connectivity.
 *
 * Handles coordinate transformations between adjacent elements with different
 * conformity types (strongly/weakly conforming, nonconforming).
 */
#pragma once

#include "dimension.hpp"
#include <string>

namespace specfem::connections {

/**
 * @brief Connection conformity types between mesh elements.
 */
enum class type : int {
  strongly_conforming = 1, ///< Nodes match exactly
  weakly_conforming = 2, ///< Nodes match, shape functions may be discontinuous
  nonconforming = 3      ///< No matching nodes, geometrically adjacent
};

/**
 * @brief Convert connection type to string.
 * @param conn Connection type
 * @return String representation
 */
const std::string to_string(const specfem::connections::type &conn);

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
template <specfem::dimension::type DimensionTag> class connection_mapping;

} // namespace specfem::connections

#include "dim2/connections.hpp"
#include "dim3/connections.hpp"

namespace specfem::connections {

/**
 * @brief Template argument deduction guides for automatic dimension detection.
 */
///@{
connection_mapping(const int, const int,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>)
    -> connection_mapping<specfem::dimension::type::dim2>;

connection_mapping(const int, const int, const int,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>,
                   Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>)
    -> connection_mapping<specfem::dimension::type::dim3>;
///@}

} // namespace specfem::connections
