#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> struct fields;

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field SimulationWavefieldType>
struct simulation_field;

} // namespace specfem::assembly

#include "fields/fields.hpp"
