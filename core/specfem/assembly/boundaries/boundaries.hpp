#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Primary template declaration for acoustic free surface boundary
 * conditions.
 */
template <specfem::element::dimension_tag DimensionTag>
struct acoustic_free_surface;

/**
 * @brief Primary template declaration for Stacey boundary conditions.
 */
template <specfem::element::dimension_tag DimensionTag> struct stacey;

} // namespace specfem::assembly::boundaries_impl

namespace specfem::assembly {

/**
 * @brief Data container for boundary condition information at every quadrature
 * point on the boundary.
 *
 * Primary template declaration. Dimension-specific partial specializations are
 * defined in boundaries/dim2/boundaries.hpp and boundaries/dim3/boundaries.hpp.
 */
template <specfem::element::dimension_tag DimensionTag> class boundaries;

} // namespace specfem::assembly
