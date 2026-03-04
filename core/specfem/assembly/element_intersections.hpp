#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Unified spectral element intersection classification and coupling
 * management
 *
 * This template provides a dimension-agnostic interface for storing and
 * managing intersection (edge or face) information in spectral element meshes.
 * In 2D the intersections are edges; in 3D they are faces. Both
 * specializations expose the same construction interface and are used
 * uniformly by conforming_interfaces.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 *
 * @see specfem::assembly::edge_types  (dim2 basis)
 * @see specfem::assembly::face_types  (dim3 basis)
 */
template <specfem::element::dimension_tag DimensionTag>
struct element_intersections;

} // namespace specfem::assembly

#include "element_intersections/dim2/element_intersections.hpp"
#include "element_intersections/dim3/element_intersections.hpp"
