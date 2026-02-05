#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element edge classification and coupling management
 *
 * This class provides storage and management for edge information in
 * spectral element meshes, including edge connectivity, interface types,
 * and boundary conditions. It handles coupling between different media
 * types and manages edge-based operations essential for discontinuous
 * Galerkin (DG) and coupled field formulations.
 *
 * The class manages edges that connect spectral elements, including:
 * - Conforming and non-conforming interfaces
 * - Elastic-acoustic coupling interfaces
 * - Free surface and absorbing boundary conditions
 * - Mortar element connections for non-matching meshes
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct edge_types;

} // namespace specfem::assembly

#include "edge_types/dim2/edge_types.hpp"
