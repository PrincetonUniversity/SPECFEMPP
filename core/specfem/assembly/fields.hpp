#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

/**
 * @brief Individual simulation field storage container
 *
 * This class provides storage and access for a specific type of simulation
 * field (forward, adjoint, backward, or buffer) in spectral element
 * computations.
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::simulation::field_type SimulationWavefieldType>
struct simulation_field;

} // namespace specfem::assembly

#include "fields/fields.hpp"
