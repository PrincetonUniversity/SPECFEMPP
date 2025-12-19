#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Individual simulation field storage container
 *
 * This class provides storage and access for a specific type of simulation
 * field (forward, adjoint, backward, or buffer) in spectral element
 * computations.
 */
template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field SimulationWavefieldType>
struct simulation_field;

} // namespace specfem::assembly

#include "fields/fields.hpp"
