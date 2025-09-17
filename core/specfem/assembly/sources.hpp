#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Assembly-level source management for spectral element simulations
 *
 * This template class manages sources within assembled finite element meshes,
 * providing efficient access to source data for both host and device
 * computations. The sources are organized by medium type (elastic, acoustic,
 * poroelastic) and support time-dependent source time functions.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag>
struct sources; ///< Forward declaration of sources class

} // namespace specfem::assembly

// Include template specializations
#include "sources/dim2/sources.hpp"
#include "sources/dim3/sources.hpp"
