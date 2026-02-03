#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/assembly.hpp"
#include "specfem/element.hpp"

namespace specfem {
namespace compute {
namespace impl {
/**
 * @brief Compute the inverse of the mass matrix for a given medium type.
 *
 * This function computes the inverse of the mass matrix for a given medium
 * type. It calls the specialized implementation for different medium types and
 * properties.
 *
 * @tparam DimensionTag Dimension of the problem
 * @tparam WavefieldType Type of the wavefield (e.g., elastic, acoustic)
 * @tparam MediumTag Medium tag (e.g., elastic, acoustic)
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::simulation::field_type WavefieldType,
          specfem::element::medium_tag MediumTag>
void invert_mass_matrix(
    const specfem::assembly::assembly<DimensionTag> &assembly);
} // namespace impl

} // namespace compute
} // namespace specfem
