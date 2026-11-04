#pragma once

#include "fk_injection/eval_points.hpp"
#include "fk_injection/field_derivative.hpp"
#include "fk_injection/fk_result.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"

namespace fk_injection {

/**
 * @brief Compute FK synthetic displacement, traction, and pressure at a set of
 *        evaluation points.
 *
 * Orchestrates two Kokkos device kernels:
 * - Kernel 1: per-frequency assembly of elastic and acoustic propagator chains
 *             and half-space coefficients (shared by all points).
 * - Kernel 2: per-point depth propagation, inverse-FFT, exponential-taper,
 *             and cubic-B-spline storage.
 *
 * @param model           1-D horizontally-layered velocity model.
 * @param wave            Incident plane-wave descriptor (type, angles, origin,
 *                        amplitude, Gaussian half-duration).
 * @param window          Simulation time-step and windowing parameters used to
 *                        derive FK working-array sizes.
 * @param points          Device-resident evaluation-point coordinates, optional
 *                        normals and Lamé factors.
 * @param derivative      Kinematic quantity stored in the result's three vector
 *                        components: displacement (Method 2), velocity (Method
 * 1 Stacey, the SPECFEM3D @c Veloc_FK convention), or acceleration (Method 2).
 * Defaults to displacement.
 * @return FkResult containing the selected kinematic field, traction (if
 *         requested), and pressure (for acoustic points) cubic-B-spline
 *         coefficient Views on the default device memory space.
 * @throws std::runtime_error on model validation failure.
 */
FkResult solve(const LayeredModel &model, const IncidentWave &wave,
               const TimeWindow &window, const EvalPoints &points,
               field_derivative derivative = field_derivative::displacement);

} // namespace fk_injection
