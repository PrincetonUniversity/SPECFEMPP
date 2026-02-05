#pragma once

#include "enumerations/interface_tags.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element type classification and indexing container
 *
 * This class provides storage and management for element type information
 * in spectral element meshes, including medium types (elastic, acoustic,
 * poroelastic), material properties (isotropic, anisotropic, Cosserat),
 * and boundary conditions.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::element::dimension_tag DimensionTag> struct element_types;

} // namespace specfem::assembly

#include "element_types/dim2/element_types.hpp"
#include "element_types/dim3/element_types.hpp"
