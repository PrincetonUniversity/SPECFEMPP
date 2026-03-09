#pragma once
#include "specfem/element.hpp"

/**
 * @brief Mesh entity definitions for spectral element connectivity.
 *
 * Provides dimension-specific entity types (edges, faces, corners) and
 * connectivity utilities for quadrilateral (2D) and hexahedral (3D) spectral
 * elements. Includes grid structures and coordinate mapping support.
 *
 */
namespace specfem::mesh_entity {

/**
 * @brief Edge entities for spectral element connectivity.
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct edge;

/**
 * @brief Element grid structure with GLL point configuration.
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct element_grid;

/**
 * @brief Element structure with coordinate mapping capabilities.
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct element;

} // namespace specfem::mesh_entity

#include "mesh_entity/dim2/mesh_entity.hpp"
#include "mesh_entity/dim3/mesh_entity.hpp"

namespace specfem::mesh_entity {

/**
 * @brief Template argument deduction guides for automatic dimension detection.
 */
///@{
element_grid(const int, const int)
    -> element_grid<specfem::element::dimension_tag::dim2>;

element_grid(const int, const int, const int)
    -> element_grid<specfem::element::dimension_tag::dim3>;

element(const int, const int) -> element<specfem::element::dimension_tag::dim2>;

element(const int, const int, const int)
    -> element<specfem::element::dimension_tag::dim3>;
///@}
} // namespace specfem::mesh_entity
