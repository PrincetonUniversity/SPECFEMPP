#pragma once

#include "specfem/element.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element face classification and coupling management
 *
 * This class provides storage and management for face information in
 * spectral element meshes, including face connectivity, interface types,
 * and boundary conditions. It handles coupling between different media
 * types and manages face-based operations essential for discontinuous
 * Galerkin (DG) and coupled field formulations in 3D.
 *
 * The class manages faces that connect spectral elements, including:
 * - Conforming and non-conforming interfaces
 * - Elastic-acoustic coupling interfaces
 * - Free surface and absorbing boundary conditions
 * - Mortar element connections for non-matching meshes
 *
 * @tparam DimensionTag Spatial dimension (3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct face_types;

} // namespace specfem::assembly

#include "face_types/dim3/face_types.hpp"
