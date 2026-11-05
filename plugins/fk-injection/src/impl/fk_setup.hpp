#pragma once

#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/register_array.hpp"
#include "impl/halfspace_solve.hpp"
#include <complex>
#include <vector>

namespace fk_injection::impl {

/**
 * @brief Frequency-domain quantities shared by every depth group.
 *
 * Built once by @c prepare from the model, wave, window and delay span. The
 * time-placement bookkeeping (t0 shift, delay pad) is added where the series
 * are synthesized; this struct holds only the frequency-domain inputs.
 */
struct FkSetup {
  fk_injection::FkSizes sizes;      ///< Working-array sizes.
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
  int pad_samples =
      0; ///< ceil(max delay / dt_min); keeps sample abscissae >= 0.
  int shift_samples = 0; ///< t0 series rotation nn - pad (left rotation), nn =
                         ///< origin_time / dt_min rounded as in @c prepare.
  double abscissa_pad = 0.0;  ///< pad_samples plus the part of origin_time /
                              ///< dt_min the rotation can't express, in [0, 1).
  double delay_stretch = 1.0; ///< Scale on every point's delay: 1, or N/(N-1)
                              ///< to reproduce SPECFEM3D's frequency grid.
  double normalization_dt = 0.0; ///< Step the inverse FFT divides by: dt_min,
                                 ///< or the SEM step to reproduce SPECFEM3D.
};

/**
 * @brief Time for the fluid block's reverberation to decay to 1e-4.
 *
 * A wave trapped between the free surface and the fluid's base loses a factor
 * \f$|R|\f$ per round trip \f$\tau = 2\sum_i H_i q_i\f$, where \f$R\f$ is the
 * fluid-to-solid P reflection coefficient at ray parameter \f$p\f$. The decay
 * time is \f$\tau \ln(10^{-4}) / \ln|R|\f$. An estimate: reflections between
 * fluid layers and reverberations in solid layers are ignored.
 *
 * @param model Layered velocity model.
 * @param p     Horizontal ray parameter (s/m).
 * @return Decay time in s; 0 without a fluid or if the wave is evanescent in
 *         it.
 * @throws std::runtime_error if the fluid's base reflects totally.
 */
double fluid_reverberation_time(const fk_injection::LayeredModel &model,
                                double p);

/**
 * @brief Build the shared frequency-domain setup for an FK synthesis.
 *
 * Sizes the working arrays to cover the time window plus the delay span,
 * and in the exact convention to at least the delay span plus
 * @c fluid_reverberation_time, so the reverberation decays within one FFT
 * period instead of wrapping onto the run. Then fills the angular-frequency and
 * Gaussian-apodization vectors, the half-space eigenmatrix, and the incident
 * amplitudes.
 *
 * @param model            Layered velocity model.
 * @param wave             Incident plane wave.
 * @param window           Time-window parameters.
 * @param p                Horizontal ray parameter from @c ray_parameter.
 * @param time_delay_min   Smallest delay of any point (s).
 * @param time_delay_max   Largest delay of any point (s).
 * @param convention       Resolved here into the frequency grid, delay stretch,
 *                         inverse-FFT normalization, origin-time truncation
 *                         and window floor; see @c ReferenceConvention.
 * @return Populated FkSetup.
 */
FkSetup prepare(const fk_injection::LayeredModel &model,
                const fk_injection::IncidentWave &wave,
                const fk_injection::TimeWindow &window, double p,
                double time_delay_min, double time_delay_max,
                fk_injection::ReferenceConvention convention =
                    fk_injection::ReferenceConvention::exact);

} // namespace fk_injection::impl
