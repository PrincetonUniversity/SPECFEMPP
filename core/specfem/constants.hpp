#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>

namespace specfem::constants {

/**
 * @brief Default path to parameter file
 *
 * This is the default path to the parameter file used by the code. It can be
 * overridden by providing a different path when running the executable.
 */
const type_real pi = 3.14159265358979323846;

/**
 * @brief Source decay rate to mimic a triangle source time function
 *
 * We mimic a triangle of half duration equal to half_duration_triangle using a
 * Gaussian having a very close shape, as explained in Figure 4.2 of the manual.
 * This source decay rate to mimic an equivalent triangle was found by trial and
 * error.
 *
 * @note From globalcmt.org: The source duration is generally estimated using an
 * empirically determined relationship such that the duration increases as the
 * cube root of the scalar moment. Specifically, we currently use a relationship
 * where the half duration for an event with moment 10**24 is 1.05 seconds, and
 * for an event with moment 10**27 is 10.5 seconds.
 *
 * @see
 * https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
 */
const type_real SOURCE_DECAY_MIMIC_TRIANGLE = 1.62800;

/**
 * @brief Suggested COURANT number for time stepping
 *
 * empirical choice for distorted elements to estimate time step and period
 * resolved: Courant number for time step estimate
 */
const type_real COURANT_NUMBER_SUGGESTED = 0.5;

/**
 * @brief Suggested number of points per minimum wavelength
 *
 * empirical choice for distorted elements to estimate time step and period
 * resolved: number of GLL points per minimum wavelength
 */
const type_real NPTS_PER_WAVELENGTH = 5;

} // namespace specfem::constants
