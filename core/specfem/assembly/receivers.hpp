#pragma once

#include "element_types.hpp"
#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "mesh.hpp"
#include "mesh/mesh.hpp"
#include "specfem/receivers.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem::assembly {

/**
 * @brief Assembly-level receiver management for spectral element simulations
 *
 * This template class manages seismic receivers within assembled finite element
 * meshes, providing efficient access to receiver data for both host and device
 * computations. The receivers support seismogram recording with various output
 * types (displacement, velocity, acceleration) and handle coordinate
 * transformations for proper seismogram orientation based on receiver geometry.
 *
 * @tparam DimensionTag The spatial dimension (dim2 or dim3)
 */
template <specfem::dimension::type DimensionTag> struct receivers;

} // namespace specfem::assembly

#include "receivers/dim2/receivers.hpp"
#include "receivers/dim3/receivers.hpp"
