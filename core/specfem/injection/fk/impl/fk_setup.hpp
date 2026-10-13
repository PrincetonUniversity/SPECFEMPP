#pragma once

#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/impl/depth_groups.hpp"
#include "specfem/injection/fk/impl/halfspace_solve.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <vector>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Frequency-domain quantities shared by every depth group.
 *
 * Built once by @c prepare from the model, wave, window and depth groups. The
 * time-placement bookkeeping (t0 shift, delay pad) is added where the series
 * are synthesized; this struct holds only the frequency-domain inputs.
 */
struct FkSetup {
  specfem::injection::fk::FkSizes sizes; ///< Working-array sizes.
  int fft_power = 0;           ///< log2(N_FFT) for the inverse transform.
  double p = 0.0;              ///< Horizontal ray parameter (s/m).
  double phi = 0.0;            ///< Azimuth (rad).
  double dt_min = 0.0;         ///< Resampled time step N_P * dt (s).
  IncidentAmplitude amplitude; ///< Half-space incident amplitudes c1 (SV), c3
                               ///< (P).
  specfem::injection::fk::incident_wave_type wave_type =
      specfem::injection::fk::incident_wave_type::p; ///< Incident wave type.
  specfem::utilities::ComplexMatrix<4> e_mat; ///< Half-space eigenmatrix E.
  std::vector<double> omega;       ///< Angular frequencies, size nf_stored+1.
  std::vector<double> apodization; ///< Gaussian STF spectrum, size nf_stored+1.
  int derivative_order = 0; ///< 0 displacement, 1 velocity, 2 acceleration.
  bool compute_traction = false; ///< Whether traction series are produced.
  int pad_samples =
      0; ///< ceil(max delay / dt_min); keeps sample abscissae >= 0.
  int shift_samples = 0; ///< t0 series rotation nn - pad (left rotation), nn =
                         ///< int(origin_time / dt_min). See doc 14.
};

/**
 * @brief Build the shared frequency-domain setup for an FK synthesis.
 *
 * Sizes the working arrays to cover the time window plus the group delay span,
 * then fills the angular-frequency and Gaussian-apodization vectors, the
 * half-space eigenmatrix, and the incident amplitudes.
 *
 * @param model            Layered velocity model.
 * @param wave             Incident plane wave.
 * @param window           Time-window parameters.
 * @param p                Horizontal ray parameter from @c ray_parameter.
 * @param groups           Depth groups (supplies the delay span for sizing).
 * @param derivative       Kinematic quantity for the vector components.
 * @param compute_traction Whether traction series are requested.
 * @return Populated FkSetup.
 */
FkSetup prepare(const specfem::injection::fk::LayeredModel &model,
                const specfem::injection::fk::IncidentWave &wave,
                const specfem::injection::fk::TimeWindow &window, double p,
                const DepthGroups &groups,
                specfem::injection::fk::field_derivative derivative,
                bool compute_traction);

} // namespace fk_impl
} // namespace injection
} // namespace specfem
