#pragma once

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Set a ComplexMatrix<4> to the identity.
 * @param M Matrix to initialise in-place.
 */
KOKKOS_INLINE_FUNCTION
void set_identity(specfem::utilities::ComplexMatrix<4> &M);

/**
 * @brief Execute the two-kernel FK synthesis pipeline.
 *
 * Called by @c specfem::injection::fk::solve() after model validation.
 * Performs all FK computation on the Kokkos default execution space.
 *
 * Kernel 1 (per-frequency, shared): assembles the elastic and acoustic
 * propagator chains and solves the free-surface half-space coefficients.
 *
 * Kernel 2 (per-point): propagates each evaluation point through its layer,
 * applies the source-time-function apodization and phase delay, inverse-FFTs
 * the five spectral field components, applies the exponential taper, and
 * stores cubic-B-spline coefficients into @c FkResult.
 *
 * @param model           Validated layered velocity model.
 * @param wave            Incident plane-wave descriptor.
 * @param window          Time-step and windowing parameters.
 * @param points          Device-resident evaluation points.
 * @param compute_traction When true, populate traction coefficient Views.
 * @param derivative      Kinematic quantity
 * (displacement/velocity/acceleration) stored in the three vector components.
 * @return Populated @c FkResult on the default device memory space.
 */
specfem::injection::fk::FkResult
run_fk_driver(const specfem::injection::fk::LayeredModel &model,
              const specfem::injection::fk::IncidentWave &wave,
              const specfem::injection::fk::TimeWindow &window,
              const specfem::injection::fk::EvalPoints &points,
              bool compute_traction,
              specfem::injection::fk::field_derivative derivative);

} // namespace fk_impl
} // namespace injection
} // namespace specfem
