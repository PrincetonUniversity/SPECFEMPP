#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
/**
 * @brief Misfit kernels (Frechet derivatives) for every quadrature point in the
 * finite element mesh
 *
 */
template <specfem::dimension::type DimensionTag> struct kernels;
} // namespace specfem::assembly

#include "kernels/dim2/kernels.hpp"
