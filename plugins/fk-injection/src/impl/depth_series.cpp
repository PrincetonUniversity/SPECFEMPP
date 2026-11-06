#include "impl/depth_series.hpp"
#include "plane_wave_response.hpp"

#include "fk_injection/utilities/cubic_bspline.hpp"
#include "fk_injection/utilities/fft.hpp"
#include "impl/field_recovery.hpp"
#include "impl/fk_math_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

void fk_injection::impl::depth_spectrum(
    const fk_injection::LayerLocation &location,
    const fk_injection::impl::InterfaceStates &states,
    const fk_injection::LayeredModel &model,
    const fk_injection::impl::FkSetup &setup, std::complex<double> *spectrum,
    int fft_length) {
  using Cplx = std::complex<double>;

  const int nf2 = static_cast<int>(setup.omega.size()); // nf_stored + 1
  const double p = setup.p;
  const bool is_acoustic =
      (location.medium == specfem::plugin::Medium::acoustic);

  for (int f = 0; f < nf2; ++f) {
    const Cplx omega(setup.omega[f], 0.0);

    const Cplx apodization(setup.apodization[f], 0.0);
    const fk_injection::Layer &model_layer = model.layers()[location.layer];
    if (is_acoustic) {
      const auto &layer = std::get<fk_injection::AcousticLayer>(model_layer);
      fk_injection::impl::AcousticState state =
          fk_injection::impl::acoustic_state_in_layer(
              layer, omega, location.height, p,
              std::get<fk_injection::impl::AcousticState>(
                  states.bottom(f, location.layer)));
      // Free-surface pressure boundary condition at the top of the fluid
      // column.
      if (location.at_free_surface) {
        state.pressure_over_k = Cplx(0.0, 0.0);
      }
      // SPECFEM3D resolves an acoustic point at a layer's bottom boundary into
      // the layer below (couple_with_injection's zz <= sum(H) tie-break), so
      // u_x = -i p^2/rho (P/k) there uses the underlying layer's density. The
      // state itself is the zero-height interface state either way.
      double horizontal_density = static_cast<double>(layer.density);
      if (location.height <= 1.0e-6 * static_cast<double>(layer.thickness)) {
        horizontal_density = std::visit(
            [](const auto &below) {
              return static_cast<double>(below.density);
            },
            model.layers()[location.layer + 1]);
      }
      // Acoustic Stacey injection consumes the incident displacement and the
      // potential rate (SPECFEM3D's acoustic FK coupling), so the kinematic
      // series stay displacement.
      auto series =
          fk_injection::impl::project_acoustic(state, horizontal_density, p);
      for (std::size_t s = 0; s < series.values.size(); ++s) {
        series.values[s] *= apodization;
        spectrum[s * fft_length + f] = series.values[s];
      }
    } else {
      fk_injection::ComplexVector<4> state;
      if (location.in_halfspace) {
        state = fk_injection::impl::elastic_state_in_halfspace(
            setup.e_mat, model.halfspace(), omega, p,
            location.depth_below_halfspace_top, states.modes_at(f));
      } else {
        state = fk_injection::impl::elastic_state_in_layer(
            std::get<fk_injection::ElasticIsotropicLayer>(model_layer), omega,
            location.height, p,
            std::get<fk_injection::ComplexVector<4>>(
                states.bottom(f, location.layer)));
      }
      // Elastic kinematic series are velocity, i omega u.
      using Series = fk_injection::ElasticMedium::Series;
      auto series = fk_injection::impl::project_elastic(state, omega, p);
      for (auto &value : series.values)
        value *= apodization;
      const Cplx i_omega(0.0, setup.omega[f]);
      series.values[Series::horizontal] *= i_omega;
      series.values[Series::vertical] *= i_omega;
      for (std::size_t s = 0; s < series.values.size(); ++s)
        spectrum[s * fft_length + f] = series.values[s];
    }
  }
}

void fk_injection::impl::to_time_series(
    std::complex<double> *spectrum, const fk_injection::impl::FkSetup &setup,
    const int number_of_series, fk_injection::type_real *coefficients) {
  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int ncoef = setup.sizes.number_of_stored_frequencies;
  const int rotation =
      ((setup.shift_samples % fft_samples) + fft_samples) % fft_samples;

  std::vector<double> real_series(fft_samples);
  std::vector<double> spline(fft_samples);

  constexpr int taper_length = 20;

  for (int s = 0; s < number_of_series; ++s) {
    fk_injection::fft::inverse_real(
        { spectrum + s * fft_samples, static_cast<std::size_t>(fft_samples) },
        setup.normalization_dt, real_series);

    if (rotation != 0) {
      std::rotate(real_series.begin(), real_series.begin() + rotation,
                  real_series.end());
    }

    fk_injection::cubic_bspline::compute_coefficients(
        real_series.data(), fft_samples, spline.data());

    // Cosine taper on the leading coefficients (suppresses inverse-FFT onset).
    const int taper = std::min(taper_length, ncoef);
    for (int k = 0; k < taper; ++k) {
      spline[k] *= 0.5 * (1.0 - std::cos(fk_injection::impl::pi *
                                         static_cast<double>(k) /
                                         static_cast<double>(taper_length)));
    }

    fk_injection::type_real *out = coefficients + s * ncoef;
    for (int k = 0; k < ncoef; ++k) {
      out[k] = static_cast<fk_injection::type_real>(spline[k]);
    }
  }
}
