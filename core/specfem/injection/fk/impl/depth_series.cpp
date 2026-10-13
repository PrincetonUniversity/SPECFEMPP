#include "specfem/injection/fk/impl/depth_series.hpp"

#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/utilities/cubic_bspline.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

void specfem::injection::fk_impl::depth_spectrum(
    const specfem::injection::fk_impl::DepthGroup &group,
    const specfem::injection::fk_impl::InterfaceStates &states,
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk_impl::FkSetup &setup,
    Kokkos::complex<double> *spectrum, int fft_length) {
  using Cplx = Kokkos::complex<double>;

  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const type_real p = static_cast<type_real>(setup.p);
  const int n = setup.derivative_order;
  const bool is_acoustic =
      (group.medium == specfem::injection::fk::point_medium::acoustic);

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    specfem::injection::fk_impl::SeriesSpectrum series;
    if (is_acoustic) {
      const specfem::injection::fk::AcousticLayer &layer =
          model.fluid_layers()[group.layer];
      specfem::injection::fk_impl::AcousticState state =
          specfem::injection::fk_impl::acoustic_state_in_layer(
              layer, omega, static_cast<type_real>(group.height), p,
              states.acoustic_bottom(f, group.layer));
      // Free-surface pressure boundary condition at the top of the fluid
      // column.
      if (group.free_surface) {
        state.pressure_over_k = Cplx(0.0, 0.0);
      }
      series =
          specfem::injection::fk_impl::project_acoustic(state, layer, omega, p);
    } else {
      specfem::utilities::ComplexVector<4> state;
      if (group.in_halfspace) {
        state = specfem::injection::fk_impl::elastic_state_in_halfspace(
            setup.e_mat, model.elastic_layers().back(), omega, p,
            group.depth_below_halfspace_top, states.modes_at(f));
      } else {
        state = specfem::injection::fk_impl::elastic_state_in_layer(
            model.elastic_layers()[group.layer], omega,
            static_cast<type_real>(group.height), p,
            states.elastic_top(f, group.layer + 1));
      }
      series = specfem::injection::fk_impl::project_elastic(
          state, omega, p, setup.compute_traction);
    }

    // Apodization on every slot; (i*omega)^n derivative on the kinematic slots.
    const Cplx apodization(setup.apodization[f], 0.0);
    Cplx derivative(1.0, 0.0);
    if (n == 1) {
      derivative = Cplx(0.0, setup.omega[f]);
    } else if (n == 2) {
      derivative = Cplx(-setup.omega[f] * setup.omega[f], 0.0);
    }
    for (int s = 0; s < specfem::injection::fk_impl::number_of_series; ++s) {
      series.values[s] *= apodization;
    }
    series.values[specfem::injection::fk_impl::series_horizontal] *= derivative;
    series.values[specfem::injection::fk_impl::series_vertical] *= derivative;

    for (int s = 0; s < specfem::injection::fk_impl::number_of_series; ++s) {
      spectrum[s * fft_length + f] = series.values[s];
    }
  }
}

void specfem::injection::fk_impl::to_time_series(
    Kokkos::complex<double> *spectrum,
    const specfem::injection::fk_impl::FkSetup &setup,
    const specfem::utilities::fft::FftTables<Kokkos::HostSpace> &tables,
    double *coefficients, bool reproduce_specfem3d_reference) {
  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const int nseries = specfem::injection::fk_impl::number_of_series;
  // Normalize by the resampled-grid step dt_min (physically correct). The
  // SPECFEM3D reference divides by the SEM step (NP times too large for NP>1);
  // reproduce that only for exact reference comparison.
  const double normalization_dt =
      reproduce_specfem3d_reference
          ? setup.dt_min / static_cast<double>(setup.sizes.resampling_rate)
          : setup.dt_min;
  const int rotation =
      ((setup.shift_samples % fft_samples) + fft_samples) % fft_samples;

  std::vector<double> real_series(fft_samples);
  std::vector<double> spline(fft_samples);

  constexpr int taper_length = 20;

  for (int s = 0; s < nseries; ++s) {
    specfem::utilities::fft::inverse(
        setup.fft_power, spectrum + s * fft_samples, -1.0, normalization_dt,
        real_series.data(), tables.bit_reversal.data(), tables.twiddles.data());

    if (rotation != 0) {
      std::rotate(real_series.begin(), real_series.begin() + rotation,
                  real_series.end());
    }

    specfem::utilities::cubic_bspline::compute_coefficients(
        real_series.data(), fft_samples, spline.data());

    double *out = coefficients + s * ncoef;
    for (int k = 0; k < ncoef; ++k) {
      out[k] = spline[k];
    }

    // Cosine taper on the leading coefficients (suppresses inverse-FFT onset).
    const int taper = std::min(taper_length, ncoef);
    for (int k = 0; k < taper; ++k) {
      const double weight =
          0.5 * (1.0 - std::cos(specfem::injection::fk_impl::pi *
                                static_cast<double>(k) /
                                static_cast<double>(taper_length)));
      out[k] *= weight;
    }
  }
}

specfem::injection::fk_impl::DepthSeries
specfem::injection::fk_impl::depth_series(
    const specfem::injection::fk_impl::DepthGroups &groups,
    const specfem::injection::fk_impl::InterfaceStates &states,
    const specfem::injection::fk::LayeredModel &model,
    const specfem::injection::fk_impl::FkSetup &setup,
    bool reproduce_specfem3d_reference) {
  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const int nseries = specfem::injection::fk_impl::number_of_series;
  const int ngroups = static_cast<int>(groups.groups.size());

  specfem::injection::fk_impl::DepthSeries result;
  result.coefficients =
      Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace>(
          "fk_depth_series", ngroups, nseries, ncoef);
  result.number_of_groups = ngroups;
  result.number_of_series = nseries;
  result.coefficient_count = ncoef;
  result.pad_samples = setup.pad_samples;
  result.shift_samples = setup.shift_samples;

  const auto tables =
      specfem::utilities::fft::make_fft_tables<Kokkos::HostSpace>(
          setup.fft_power, -1.0);

  std::vector<Kokkos::complex<double>> spectrum(
      static_cast<std::size_t>(nseries) * fft_samples);

  for (int g = 0; g < ngroups; ++g) {
    std::fill(spectrum.begin(), spectrum.end(),
              Kokkos::complex<double>(0.0, 0.0));
    specfem::injection::fk_impl::depth_spectrum(
        groups.groups[g], states, model, setup, spectrum.data(), fft_samples);
    specfem::injection::fk_impl::to_time_series(spectrum.data(), setup, tables,
                                                &result.coefficients(g, 0, 0),
                                                reproduce_specfem3d_reference);
  }

  return result;
}
