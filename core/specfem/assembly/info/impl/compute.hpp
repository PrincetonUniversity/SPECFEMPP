#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::info::impl {

/**
 * @brief Compute average GLL point spacing from element size.
 *
 * @param element_size The maximum distance across an element (edge length).
 * @param ngll_minus_one Number of GLL points minus one (polynomial order).
 * @return Average distance between adjacent GLL points.
 */
KOKKOS_INLINE_FUNCTION
type_real compute_average_gll_spacing(type_real element_size,
                                      int ngll_minus_one) {
  return element_size / static_cast<type_real>(ngll_minus_one);
}

/**
 * @brief Compute the minimum resolvable period for an element.
 *
 * The minimum period that can be accurately resolved depends on the
 * spatial sampling (points per wavelength) and the wave velocity.
 * Uses the empirical constant NPTS_PER_WAVELENGTH.
 *
 * @param avg_gll_spacing Average distance between GLL points.
 * @param min_velocity Minimum wave velocity in the element.
 * @return Minimum resolvable period (wavelength / velocity).
 */
KOKKOS_INLINE_FUNCTION
type_real compute_minimum_period(type_real avg_gll_spacing,
                                 type_real min_velocity) {
  return (specfem::constants::empirical::NPTS_PER_WAVELENGTH * avg_gll_spacing) /
         min_velocity;
}

/**
 * @brief Compute suggested time step based on CFL condition.
 *
 * The Courant-Friedrichs-Lewy (CFL) condition ensures numerical stability
 * by limiting how far a wave can travel in one time step.
 * Uses the empirical constant COURANT_NUMBER_SUGGESTED.
 *
 * @param min_gll_distance Minimum distance between adjacent GLL points.
 * @param max_velocity Maximum wave velocity in the element.
 * @return Suggested time step satisfying the CFL condition.
 */
KOKKOS_INLINE_FUNCTION
type_real compute_suggested_timestep(type_real min_gll_distance,
                                     type_real max_velocity) {
  return specfem::constants::empirical::COURANT_NUMBER_SUGGESTED *
         (min_gll_distance / max_velocity);
}

} // namespace specfem::assembly::info::impl
