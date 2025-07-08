#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

/**
 * @brief Material properties at every quadrature point in the finite element
 * mesh
 *
 */
template <specfem::dimension::type DimensionTag> struct properties;
} // namespace specfem::assembly

#include "properties/dim2/properties.hpp"
