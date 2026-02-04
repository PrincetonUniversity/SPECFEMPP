#pragma once
#include "specfem/element.hpp"

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

#include "dim2/mesh_entities.hpp"
#include "dim3/mesh_entities.hpp"

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
