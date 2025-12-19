#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Data container for misfit kernels within the spectral element domain
 *
 * This class provides storage and data access functions for the misfit kernels
 * associated with spectral elements in spectral element mesh. The dimension
 * specific implementations provide data containers for storing misfit kernels
 * as well as methods for loading and storing data on device and host.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 */
template <specfem::dimension::type DimensionTag> struct kernels;

} // namespace specfem::assembly

#include "kernels/dim2/kernels.hpp"
#include "kernels/dim3/kernels.hpp"

// Device-side data access functions for GPU kernels
#include "kernels/data_access/add_on_device.hpp" ///< Accumulate kernel data on GPU device
#include "kernels/data_access/load_on_device.hpp" ///< Load kernel data on GPU device
#include "kernels/data_access/store_on_device.hpp" ///< Store kernel data on GPU device

// Host-side data access functions for CPU operations
#include "kernels/data_access/add_on_host.hpp" ///< Accumulate kernel data on CPU host
#include "kernels/data_access/load_on_host.hpp" ///< Load kernel data on CPU host
#include "kernels/data_access/store_on_host.hpp" ///< Store kernel data on CPU host

/** @} */ // end of KernelsDataAccess group
