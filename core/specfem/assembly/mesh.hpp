#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::mesh_impl {
template <specfem::element::dimension_tag Dimension> struct control_nodes;

template <specfem::element::dimension_tag Dimension>
struct mesh_to_compute_mapping;

template <specfem::element::dimension_tag Dimension> struct points;

template <specfem::element::dimension_tag Dimension> struct shape_functions;

template <specfem::element::dimension_tag Dimension> struct quadrature;

template <specfem::element::dimension_tag Dimension> struct adjacency_graph;
} // namespace specfem::assembly::mesh_impl

namespace specfem::assembly {
/**
 * @brief Assembled 3D mesh representation for spectral element analysis.
 *
 * @tparam Dimension The spatial dimension (e.g., dim2 or dim3).
 */
template <specfem::element::dimension_tag Dimension> struct mesh;
} // namespace specfem::assembly

// Include dim2 declarations
#include "mesh/dim2/mesh.hpp"

// Include dim3 declarations
#include "mesh/dim3/mesh.hpp"

// Include data access implementations
#include "mesh/data_access.hpp"
