#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief One medium's simulation field, restricted to a range of elements.
 *
 * Returned by `specfem::assembly::subview(field, range)`. Elements are
 * numbered from 0 at the start of the range.
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam SimulationWavefieldType Wavefield of the parent field
 * @tparam MediumTag Medium the range covers
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::simulation::field_type SimulationWavefieldType,
          specfem::element::medium_tag MediumTag>
struct SimulationFieldSubview;

} // namespace specfem::assembly

#include "subview/dim3/simulation_field.hpp"
