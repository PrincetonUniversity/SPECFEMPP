#pragma once

#include "specfem/setup.hpp"

namespace specfem {
namespace injection {
namespace fk {

/**
 * @brief Time-domain sampling parameters for the FK computation.
 *
 * Carries the simulation time step and windowing parameters that
 * @c compute_fk_sizes() uses to determine the working-array sizes.
 */
struct TimeWindow {
  type_real dt = 0;            ///< Simulation time step in s
  int nstep = 0;               ///< Total number of simulation time steps
  type_real frequency_max = 0; ///< Maximum frequency of interest in Hz
  type_real frequency_sampling =
      0; ///< Nyquist-like sampling frequency in Hz (\f$F_\text{ech}\f$)
  type_real time_window_length =
      0; ///< Requested FK time-window length in s (\f$T_\text{max}\f$)
};

/**
 * @brief Working-array sizes derived from a @c TimeWindow.
 *
 * Returned by @c compute_fk_sizes(); all integer sizes are exact powers of two
 * where applicable.
 */
struct FkSizes {
  int number_of_stored_frequencies =
      0; ///< \f$N_{F,\text{store}}\f$ — number of frequency samples stored
         ///< (power of 2)
  int number_of_fft_samples = 0; ///< \f$N_\text{FFT}\f$ — total FFT length (= 2
                                 ///< * number_of_stored_frequencies)
  int interpolation_power =
      0; ///< \f$N_\text{pow,interp}\f$ — power-of-2 exponent for interpolation
         ///< (\f$\log_2 N_\text{FFT}\f$)
  int resampling_rate = 0; ///< \f$N_{P,\text{resamp}}\f$ — ratio of resampled
                           ///< to simulation time step
  type_real frequency_step =
      0; ///< \f$\Delta f\f$ in Hz (\f$= 1/((N_\text{FFT}-1)\Delta
         ///< t_\text{min})\f$; follows the SPECFEM3D reference, which stretches
         ///< frequency-domain delays by a factor \f$N/(N-1)\f$)
  type_real effective_time_window =
      0; ///< Effective FK time window \f$T_\text{FK} = \Delta
         ///< t_\text{min}(N_\text{FFT}-1)\f$ in s
};

/**
 * @brief Compute working-array sizes from a @c TimeWindow.
 *
 * Pure function; no side-effects.  Ports @c find_size_of_working_arrays from
 * the Fortran reference, with @c nstep as an explicit field instead of a
 * hidden module global.
 *
 * Algorithm:
 * -# Compute the minimum resampled time step
 *    \f$ \Delta t_\text{min} = N_{P,\text{resamp}} \, \Delta t \f$.
 * -# Compute the usable time window
 *    \f$ T_\text{use} = \min(T_\text{max,samp},\, T_\text{window} +
 *    T_\text{delay\_span}) \f$.
 * -# Round up to the next power of two for the stored-frequency count.
 * -# Set FFT length to twice that count; frequency step to
 *    \f$ \Delta f = 1 / T_\text{FK} \f$.
 *
 * @param window           Simulation time and windowing parameters.
 * @param time_delay_span  Extra time in seconds added to the requested window
 *                         before the power-of-two rounding; used to accommodate
 *                         FK plane-wave time delays (default 0).
 * @return Populated @c FkSizes.
 * @throws std::runtime_error if @c window.frequency_sampling is not positive.
 */
FkSizes compute_fk_sizes(const TimeWindow &window, double time_delay_span = 0);

} // namespace fk
} // namespace injection
} // namespace specfem
