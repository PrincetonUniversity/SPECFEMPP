#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
/**
 * @brief Element types for every quadrature point in the
 * finite element mesh
 *
 */
template <specfem::dimension::type DimensionTag> struct element_types;

} // namespace specfem::assembly

#include "element_types/dim2/element_types.hpp"
#include "element_types/dim3/element_types.hpp"
