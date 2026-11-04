#pragma once

#include "fk_injection/field_derivative.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/depth_groups.hpp"
#include "impl/halfspace_solve.hpp"
#include <complex>
#include <vector>

namespace fk_injection::impl {

/**
 * @brief Frequency-domain quantities shared by every depth group.
 *
 * Built once by @c prepare from the model, wave, window and depth groups. The
 * time-placement bookkeeping (t0 shift, delay pad) is added where the series
 * are synthesized; this struct holds only the frequency-domain inputs.
 */
struct FkSetup {
  fk_injection::FkSizes sizes;      ///< Working-array sizes.
  int fft_power = 0;                ///< log2(N_FFT) for the inverse transform.
  double p = 0.0;                   ///< Horizontal ray parameter (s/m).
  double phi = 0.0;                 ///< Azimuth (rad).
  double dt_min = 0.0;              ///< Resampled time step N_P * dt (s).
  std::complex<double> amplitude{}; ///< Incident mode amplitude in the
                                    ///< half-space.
  fk_injection::incident_wave_type wave_type =
      fk_injection::incident_wave_type::p; ///< Incident wave type.
  fk_injection::ComplexMatrix<4> e_mat;    ///< Half-space eigenmatrix E.
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
FkSetup prepare(const fk_injection::LayeredModel &model,
                const fk_injection::IncidentWave &wave,
                const fk_injection::TimeWindow &window, double p,
                const DepthGroups &groups,
                fk_injection::field_derivative derivative,
                bool compute_traction);

} // namespace fk_injection::impl
