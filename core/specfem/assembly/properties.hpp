#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Spectral element material properties data container
 *
 * This class provides storage and data access functions for the material
 * properties associated with spectral elements in spectral element mesh. The
 * dimension specific implementations provide data containers for storing
 * material properties such as density, elastic constants, and other medium
 * parameters as well as methods for loading and storing data on device and
 * host.
 *
 */
template <specfem::element::dimension_tag DimensionTag> struct properties;

} // namespace specfem::assembly

#include "properties/dim2/properties.hpp"
#include "properties/dim3/properties.hpp"

// Data access functions
#include "properties/load_on_device.hpp"
#include "properties/load_on_host.hpp"
#include "properties/store_on_host.hpp"
