#include "impl/depth_series.hpp"

#include "fk_injection/utilities/cubic_bspline.hpp"
#include "fk_injection/utilities/fft.hpp"
#include "impl/field_recovery.hpp"
#include "impl/fk_math_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void fk_injection::impl::depth_spectrum(
    const fk_injection::impl::DepthGroup &group,
    const fk_injection::impl::InterfaceStates &states,
    const fk_injection::LayeredModel &model,
    const fk_injection::impl::FkSetup &setup, std::complex<double> *spectrum,
    int fft_length) {
  using Cplx = std::complex<double>;

  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const double p = setup.p;
  const bool is_acoustic = (group.medium == specfem::plugin::Medium::acoustic);

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    fk_injection::impl::SeriesSpectrum series;
    if (is_acoustic) {
      const fk_injection::AcousticLayer &layer =
          model.fluid_layers()[group.layer];
      fk_injection::impl::AcousticState state =
          fk_injection::impl::acoustic_state_in_layer(
              layer, omega, group.height, p,
              states.acoustic_bottom(f, group.layer));
      // Free-surface pressure boundary condition at the top of the fluid
      // column.
      if (group.free_surface) {
        state.pressure_over_k = Cplx(0.0, 0.0);
      }
      // SPECFEM3D resolves an acoustic point at a layer's bottom boundary into
      // the layer below (couple_with_injection's zz <= sum(H) tie-break), so
      // u_x = -i p^2/rho (P/k) there uses the underlying layer's density. The
      // state itself is the zero-height interface state either way.
      double horizontal_density = static_cast<double>(layer.density);
      if (group.height <= 1.0e-6 * static_cast<double>(layer.thickness)) {
        const int below = group.layer + 1;
        horizontal_density =
            below < model.number_of_fluid_layers()
                ? static_cast<double>(model.fluid_layers()[below].density)
                : static_cast<double>(model.elastic_layers().front().density);
      }
      series =
          fk_injection::impl::project_acoustic(state, horizontal_density, p);
    } else {
      fk_injection::ComplexVector<4> state;
      if (group.in_halfspace) {
        state = fk_injection::impl::elastic_state_in_halfspace(
            setup.e_mat, model.elastic_layers().back(), omega, p,
            group.depth_below_halfspace_top, states.modes_at(f));
      } else {
        state = fk_injection::impl::elastic_state_in_layer(
            model.elastic_layers()[group.layer], omega, group.height, p,
            states.elastic_top(f, group.layer + 1));
      }
      series = fk_injection::impl::project_elastic(state, omega, p);
    }

    // Apodization on every slot; i*omega for velocity on the kinematic slots.
    // Acoustic Stacey injection consumes the incident displacement and the
    // potential rate (SPECFEM3D's acoustic FK coupling), so acoustic kinematic
    // series stay displacement.
    const Cplx apodization(setup.apodization[f], 0.0);
    Cplx derivative(1.0, 0.0);
    if (!is_acoustic &&
        setup.derivative == fk_injection::field_derivative::velocity) {
      derivative = Cplx(0.0, setup.omega[f]);
    }
    for (int s = 0; s < fk_injection::impl::number_of_series; ++s) {
      series.values[s] *= apodization;
    }
    series.values[fk_injection::impl::series_horizontal] *= derivative;
    series.values[fk_injection::impl::series_vertical] *= derivative;

    for (int s = 0; s < fk_injection::impl::number_of_series; ++s) {
      spectrum[s * fft_length + f] = series.values[s];
    }
  }
}

void fk_injection::impl::to_time_series(
    std::complex<double> *spectrum, const fk_injection::impl::FkSetup &setup,
    double *coefficients, fk_injection::ReferenceConvention convention) {
  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const int nseries = fk_injection::impl::number_of_series;
  // Normalize by the resampled-grid step dt_min (physically correct). The
  // SPECFEM3D reference divides by the SEM step (NP times too large for NP>1);
  // reproduce that only for exact reference comparison.
  const double normalization_dt =
      convention == fk_injection::ReferenceConvention::specfem3d
          ? setup.dt_min / static_cast<double>(setup.sizes.resampling_rate)
          : setup.dt_min;
  const int rotation =
      ((setup.shift_samples % fft_samples) + fft_samples) % fft_samples;

  std::vector<double> real_series(fft_samples);
  std::vector<double> spline(fft_samples);

  constexpr int taper_length = 20;

  for (int s = 0; s < nseries; ++s) {
    fk_injection::fft::inverse_real(
        { spectrum + s * fft_samples, static_cast<std::size_t>(fft_samples) },
        normalization_dt, real_series);

    if (rotation != 0) {
      std::rotate(real_series.begin(), real_series.begin() + rotation,
                  real_series.end());
    }

    fk_injection::cubic_bspline::compute_coefficients(
        real_series.data(), fft_samples, spline.data());

    double *out = coefficients + s * ncoef;
    for (int k = 0; k < ncoef; ++k) {
      out[k] = spline[k];
    }

    // Cosine taper on the leading coefficients (suppresses inverse-FFT onset).
    const int taper = std::min(taper_length, ncoef);
    for (int k = 0; k < taper; ++k) {
      const double weight =
          0.5 *
          (1.0 - std::cos(fk_injection::impl::pi * static_cast<double>(k) /
                          static_cast<double>(taper_length)));
      out[k] *= weight;
    }
  }
}

fk_injection::impl::DepthSeries fk_injection::impl::depth_series(
    const fk_injection::impl::DepthGroups &groups,
    const fk_injection::impl::InterfaceStates &states,
    const fk_injection::LayeredModel &model,
    const fk_injection::impl::FkSetup &setup,
    fk_injection::ReferenceConvention convention) {
  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const int nseries = fk_injection::impl::number_of_series;
  const int ngroups = static_cast<int>(groups.groups.size());

  fk_injection::impl::DepthSeries result;
  result.coefficients.resize(static_cast<std::size_t>(ngroups) * nseries *
                             ncoef);
  result.number_of_groups = ngroups;
  result.coefficient_count = ncoef;

  std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(nseries) *
                                             fft_samples);

  for (int g = 0; g < ngroups; ++g) {
    std::fill(spectrum.begin(), spectrum.end(), std::complex<double>(0.0, 0.0));
    fk_injection::impl::depth_spectrum(groups.groups[g], states, model, setup,
                                       spectrum.data(), fft_samples);
    fk_injection::impl::to_time_series(
        spectrum.data(), setup, &result.coefficients[result.index(g, 0, 0)],
        convention);
  }

  return result;
}
