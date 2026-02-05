#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element Jacobian matrix data container
 *
 * This class provides storage and data access functions for the Jacobian
 * matrices associated with spectral elements in spectral element mesh. The
 * dimension specific implementations provide data containers for storing
 * individual terms of the Jacobian matrix as well as methods for loading and
 * storing data on device and host.
 *
 */
template <specfem::element::dimension_tag DimensionTag> struct jacobian_matrix;

} // namespace specfem::assembly

// Include dimension-specific implementations
#include "jacobian_matrix/dim2/jacobian_matrix.hpp"
#include "jacobian_matrix/dim3/jacobian_matrix.hpp"

// Data access functions
#include "jacobian_matrix/load_on_device.hpp"
#include "jacobian_matrix/load_on_host.hpp"
#include "jacobian_matrix/store_on_device.hpp"
#include "jacobian_matrix/store_on_host.hpp"
