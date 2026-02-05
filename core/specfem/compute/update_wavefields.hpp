#pragma once

#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"

namespace specfem::compute {
/**
 * @brief Updates the wavefield for a given medium
 *
 * This function updates the wavefield for a given medium type. It computes
 * the coupling, source interaction, stiffness interaction, and divides the
 * mass matrix. The function is specialized for different medium types and
 * properties.
 *
 * @tparam WavefieldType Type of the wavefield
 * @tparam DimensionTag Dimension tag
 * @tparam NGLL Number of GLL points
 * @tparam MediumTag Medium for which the wacefield is updated
 * @param assembly The assembly object containing the mesh
 * @param istep Time step for which the wavefield is updated
 * @return int Number of elements updated
 */
template <specfem::simulation::field_type WavefieldType,
          specfem::element::dimension_tag DimensionTag, int NGLL,
          specfem::element::medium_tag MediumTag>
int update_wavefields(specfem::assembly::assembly<DimensionTag> &assembly,
                      const int istep);
} // namespace specfem::compute
