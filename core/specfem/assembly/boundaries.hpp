#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Data required to compute acoustic free surface boundary conditions
 *
 * General template declaration for acoustic free surface boundary conditions.
 *
 * @see
 * specfem::assembly::boundaries_impl::acoustic_free_surface<specfem::element::dimension_tag::dim2>
 * for 2D specialization.
 * @see
 * specfem::assembly::boundaries_impl::acoustic_free_surface<specfem::element::dimension_tag::dim3>
 * for 3D specialization.
 */
template <specfem::element::dimension_tag DimensionTag>
struct acoustic_free_surface;

/**
 * @brief Data required to compute Stacey boundary conditions
 *
 * General template declaration for Stacey boundary conditions.
 *
 */
template <specfem::element::dimension_tag DimensionTag> struct stacey;

} // namespace specfem::assembly::boundaries_impl

namespace specfem::assembly {

/**
 * @brief Data container used to store information required to implement
 * boundary conditions at every quadrature point on the boundary
 *
 * General template declaration for boundary condition information. Individual
 * template specializations for 2D and 3D provide specialized containers that
 * store data required for different types of boundary conditions (e.g.,
 * acoustic free surface, Stacey boundary conditions, etc.)
 */
template <specfem::element::dimension_tag DimensionTag> class boundaries;

} // namespace specfem::assembly

#include "boundaries/dim2/boundaries.hpp"
#include "boundaries/dim3/boundaries.hpp"
