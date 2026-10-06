#include "specfem/injection/fk/time_window.hpp"
#include <cmath>

specfem::injection::fk::FkSizes
specfem::injection::fk::compute_fk_sizes(const TimeWindow &window) {
  FkSizes sizes;

  // Resampling rate and minimum resampled time step.
  const type_real Frq_ech = window.frequency_sampling;
  const type_real dt_min_raw = static_cast<type_real>(1) / Frq_ech;

  int NP_RESAMP = static_cast<int>(std::floor(dt_min_raw / window.dt));
  if (NP_RESAMP == 0) {
    NP_RESAMP = 1;
  }
  const type_real dt_min = static_cast<type_real>(NP_RESAMP) * window.dt;

  // Usable sampled time window (integer division mirrors the Fortran
  // reference).
  const type_real tmax_samp =
      static_cast<type_real>(2 * window.nstep / NP_RESAMP + 1) * dt_min;
  type_real tmax_use = window.time_window_length;
  if (tmax_use > tmax_samp) {
    tmax_use = tmax_samp;
  }

  // Round up stored-frequency count to a power of two.
  int NF_FOR_STORING = static_cast<int>(std::ceil(tmax_use / dt_min));
  NF_FOR_STORING = static_cast<int>(
      std::ceil(std::log(static_cast<type_real>(NF_FOR_STORING)) /
                std::log(static_cast<type_real>(2))));
  // NF_FOR_STORING is now a power-of-2 exponent.

  const int NF_FOR_FFT = 1 << (NF_FOR_STORING + 1);
  const int NPOW_FOR_INTERP = NF_FOR_STORING + 1;
  NF_FOR_STORING = 1 << NF_FOR_STORING;

  const type_real tmax_fk = dt_min * static_cast<type_real>(NF_FOR_FFT - 1);
  const type_real DF_FK = static_cast<type_real>(1) / tmax_fk;

  sizes.number_of_stored_frequencies = NF_FOR_STORING;
  sizes.number_of_fft_samples = NF_FOR_FFT;
  sizes.interpolation_power = NPOW_FOR_INTERP;
  sizes.resampling_rate = NP_RESAMP;
  sizes.fft_power = 0; // set by the driver
  sizes.frequency_step = DF_FK;
  sizes.effective_time_window = tmax_fk;

  return sizes;
}
