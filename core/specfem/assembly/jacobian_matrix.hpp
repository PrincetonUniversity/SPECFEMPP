#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
/**
 * @brief Jacobian matrix of the basis functions at every quadrature point
 *
 */
template <specfem::dimension::type DimensionTag> struct jacobian_matrix;

} // namespace specfem::assembly

#include "jacobian_matrix/dim2/jacobian_matrix.hpp"
