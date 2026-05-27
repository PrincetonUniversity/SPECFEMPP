#pragma once

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <memory>
#include <vector>

namespace specfem {
namespace io {
namespace sources_impl {

/**
 * @brief Adjust t0 and tshift across all sources.
 *
 * Computes the minimum t0 and tshift across all sources, then either validates
 * the user-defined start time or auto-adjusts tshift values accordingly.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param sources Vector of source objects (tshift may be modified in-place)
 * @param user_t0 User-defined start time (0 means auto-detect)
 * @return type_real The computed or validated t0
 */
template <specfem::element::dimension_tag DimensionTag>
type_real adjust_source_timing(
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources,
    type_real user_t0);

/**
 * @brief Validate adjoint source counts against simulation type.
 *
 * Counts adjoint sources by checking wavefield type, then validates that
 * combined simulations have adjoint sources and forward simulations do not.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param sources Vector of source objects
 * @param simulation_type The simulation type (forward, combined, etc.)
 */
template <specfem::element::dimension_tag DimensionTag>
void validate_source_simulation_type(
    const std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources,
    specfem::simulation::type simulation_type);

} // namespace sources_impl
} // namespace io
} // namespace specfem

#include "timing.tpp"
