#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Container for spectral element Jacobian matrix components.
 *
 * Stores Jacobian matrix terms for coordinate transformations between
 * reference and physical element coordinates. Dimension-specific
 * specializations manage component data on both host and device.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 *
 * @ingroup Assembly
 */
template <specfem::element::dimension_tag DimensionTag> struct jacobian_matrix;

} // namespace specfem::assembly

// Include dimension-specific implementations
#include "jacobian_matrix/dim2/jacobian_matrix.hpp"
#include "jacobian_matrix/dim3/jacobian_matrix.hpp"

// Data access functions
#include "jacobian_matrix/load_on_device.hpp"
#include "jacobian_matrix/load_on_host.hpp"
#include "jacobian_matrix/prefetch_on_device.hpp"
#include "jacobian_matrix/prefetch_on_host.hpp"
#include "jacobian_matrix/store_on_device.hpp"
#include "jacobian_matrix/store_on_host.hpp"
