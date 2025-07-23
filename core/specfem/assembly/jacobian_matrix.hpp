#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
/**
 * @brief Jacobian matrix of the basis functions at every quadrature point
 *
 */
template <specfem::dimension::type DimensionTag> struct jacobian_matrix;

} // namespace specfem::assembly

// Include dimension-specific implementations
#include "jacobian_matrix/dim2/jacobian_matrix.hpp"
#include "jacobian_matrix/dim3/jacobian_matrix.hpp"

// Data access functions
#include "jacobian_matrix/dim3/load_on_device.hpp"
#include "jacobian_matrix/dim3/load_on_host.hpp"
#include "jacobian_matrix/dim3/store_on_host.hpp"
