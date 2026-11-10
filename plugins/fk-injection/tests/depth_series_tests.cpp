#include "fk_injection/config.hpp"
#include "fk_injection/eval_points.hpp"
#include "fk_injection/field_derivative.hpp"
#include "fk_injection/fk_result.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/solver.hpp"
#include "fk_injection/time_window.hpp"
#include "impl/depth_groups.hpp"
#include "impl/depth_series.hpp"
#include "impl/driver.hpp"
#include "impl/field_recovery.hpp"
#include "impl/fk_setup.hpp"
#include "impl/layer_stack.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using fk_injection::type_real;

// ---------------------------------------------------------------------------
// Helpers in a named namespace (no anonymous namespaces; unity-build safe).
// ---------------------------------------------------------------------------
namespace depth_series_tests_impl {

/** @brief Build an EvalPoints from parallel arrays. */
fk_injection::EvalPoints make_eval_points(const std::vector<double> &xs,
                                          const std::vector<double> &ys,
                                          const std::vector<double> &zs) {
  const int n = static_cast<int>(xs.size());

  fk_injection::EvalPoints ep;
  ep.x.resize(n);
  ep.y.resize(n);
  ep.z.resize(n);
  for (int i = 0; i < n; ++i) {
    ep.x[i] = static_cast<type_real>(xs[i]);
    ep.y[i] = static_cast<type_real>(ys[i]);
    ep.z[i] = static_cast<type_real>(zs[i]);
  }

  return ep;
}

/** @brief Elastic-only model: one finite layer (H=20000) + halfspace. */
fk_injection::LayeredModel make_elastic_model() {
  fk_injection::ElasticIsotropicLayer finite_layer;
  finite_layer.density = static_cast<type_real>(2800.0);
  finite_layer.p_velocity = static_cast<type_real>(6000.0);
  finite_layer.s_velocity = static_cast<type_real>(3464.0);
  finite_layer.thickness = static_cast<type_real>(20000.0);

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = static_cast<type_real>(3380.0);
  halfspace.p_velocity = static_cast<type_real>(8100.0);
  halfspace.s_velocity = static_cast<type_real>(4500.0);
  halfspace.thickness = static_cast<type_real>(0.0);

  return fk_injection::LayeredModel({}, { finite_layer, halfspace });
}

/** @brief Incident P wave: theta=25 deg, phi=0, origin_time=-5.05 s. */
fk_injection::IncidentWave make_wave_p() {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0.0);
  wave.take_off_theta =
      static_cast<type_real>(25.0 * 3.141592653589793 / 180.0);
  wave.origin_x = static_cast<type_real>(0.0);
  wave.origin_y = static_cast<type_real>(0.0);
  wave.origin_z = static_cast<type_real>(0.0);
  wave.origin_time = static_cast<type_real>(-5.05);
  wave.amplitude = static_cast<type_real>(1.0);
  wave.gaussian_half_duration = static_cast<type_real>(0.0);
  return wave;
}

/** @brief Standard time window (dt=0.05 s, 512 steps, fmax=1 Hz). */
fk_injection::TimeWindow make_window() {
  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 512;
  window.frequency_max = static_cast<type_real>(1.0);
  window.frequency_sampling = static_cast<type_real>(4.0);
  window.time_window_length = static_cast<type_real>(50.0);
  return window;
}

/**
 * @brief Build the full FK chain up to InterfaceStates.
 *
 * Returns (groups, setup, states) for three evaluation points at depths
 * 5000, 15000, and 25000 m; the shallowest (5000 m) falls inside the
 * finite layer and has in_halfspace == false.
 */
struct FkChain {
  fk_injection::impl::DepthGroups groups;
  fk_injection::impl::FkSetup setup;
  fk_injection::impl::InterfaceStates states;
};

FkChain make_fk_chain() {
  const auto model = make_elastic_model();
  const auto wave = make_wave_p();
  const auto window = make_window();

  // Three depths: one inside the finite layer, one near the layer boundary,
  // one inside the halfspace.
  const std::vector<double> zs = { 5000.0, 15000.0, 25000.0 };
  const int n = static_cast<int>(zs.size());
  const std::vector<double> xs(n, 0.0);
  const std::vector<double> ys(n, 0.0);

  const auto points = make_eval_points(xs, ys, zs);

  const double p = wave.ray_parameter(model);
  const double eps = fk_injection::impl::depth_tolerance(25000.0);
  const auto groups =
      fk_injection::impl::group_by_depth(points, model, wave, p, eps);

  const auto setup =
      fk_injection::impl::prepare(model, wave, window, p, groups,
                                  fk_injection::field_derivative::displacement);

  const auto states = fk_injection::impl::solve_layer_stack(model, setup);

  return FkChain{ groups, setup, states };
}

} // namespace depth_series_tests_impl

// ============================================================================
// Test 1: SpectrumIdentityDisplacement
//
// For displacement mode the identity i*omega*p*S0(f) == S4(f) must hold at
// every stored positive frequency for a non-halfspace group.
// ============================================================================
TEST(DepthSeries, SpectrumIdentityDisplacement) {
  const auto model = depth_series_tests_impl::make_elastic_model();
  const auto chain = depth_series_tests_impl::make_fk_chain();
  const auto &groups = chain.groups;
  const auto &setup = chain.setup;
  const auto &states = chain.states;

  // Find the first non-halfspace group (depth 5000 m is inside the finite
  // layer and must satisfy in_halfspace == false).
  int target_group = -1;
  for (int g = 0; g < static_cast<int>(groups.groups.size()); ++g) {
    if (!groups.groups[g].in_halfspace) {
      target_group = g;
      break;
    }
  }
  ASSERT_GE(target_group, 0) << "no non-halfspace group found; add shallower "
                                "points to make_fk_chain()";

  const fk_injection::impl::DepthGroup &group = groups.groups[target_group];

  const int nf_total = setup.sizes.number_of_fft_samples;
  const int nf_stored = setup.sizes.number_of_stored_frequencies;

  std::vector<std::complex<double>> spec(fk_injection::impl::number_of_series *
                                             nf_total,
                                         std::complex<double>(0.0, 0.0));

  fk_injection::impl::depth_spectrum(group, states, model, setup, spec.data(),
                                     nf_total);

  // Check that the group is non-trivial: max |S4| over positive frequencies
  // should exceed 1e-6.
  double max_s4 = 0.0;
  for (int f = 1; f <= nf_stored; ++f) {
    const double s4_abs =
        std::abs(spec[fk_injection::impl::series_dx * nf_total + f]);
    if (s4_abs > max_s4)
      max_s4 = s4_abs;
  }
  EXPECT_GT(max_s4, 1.0e-6)
      << "S4 is uniformly tiny; the group appears degenerate";

  // Identity: i * omega * p * S0 == S4.
  const double p_val = setup.p;
  for (int f = 1; f <= nf_stored; ++f) {
    const std::complex<double> s0 =
        spec[fk_injection::impl::series_horizontal * nf_total + f];
    const std::complex<double> s4 =
        spec[fk_injection::impl::series_dx * nf_total + f];

    const std::complex<double> lhs =
        std::complex<double>(0.0, setup.omega[f] * p_val) * s0;

    const double abs_diff = std::abs(lhs - s4);
    const double scale = std::abs(s4) + 1.0e-30;

    // Pass when either the relative error is small or both sides are tiny.
    EXPECT_TRUE(abs_diff <= 1.0e-9 * scale || abs_diff < 1.0e-12)
        << "Identity i*omega*p*S0 == S4 violated at frequency index f=" << f
        << ": lhs=" << lhs << " s4=" << s4 << " abs_diff=" << abs_diff
        << " scale=" << scale;
  }
}

// ============================================================================
// Test 2: ShiftAndPadMatchFormula
//
// pad_samples == ceil(time_delay_max / dt_min)
// shift_samples == int(origin_time / dt_min) - pad_samples
// ============================================================================
TEST(DepthSeries, ShiftAndPadMatchFormula) {
  const auto chain = depth_series_tests_impl::make_fk_chain();
  const auto &groups = chain.groups;
  const auto &setup = chain.setup;
  const auto &wave = depth_series_tests_impl::make_wave_p();

  const double dt_min = setup.dt_min;
  ASSERT_GT(dt_min, 0.0) << "dt_min must be positive";

  const int expected_pad =
      static_cast<int>(std::ceil(groups.time_delay_max / dt_min));
  const int expected_nn =
      static_cast<int>(static_cast<double>(wave.origin_time) / dt_min);
  const int expected_shift = expected_nn - expected_pad;

  EXPECT_EQ(setup.pad_samples, expected_pad)
      << "pad_samples mismatch: got " << setup.pad_samples << " expected "
      << expected_pad;
  EXPECT_EQ(setup.shift_samples, expected_shift)
      << "shift_samples mismatch: got " << setup.shift_samples << " expected "
      << expected_shift;
}

// ============================================================================
// Test 3: CoefficientsFiniteAndNonzero
//
// depth_series() must produce finite, non-degenerate B-spline coefficients.
// ============================================================================
TEST(DepthSeries, CoefficientsFiniteAndNonzero) {
  const auto model = depth_series_tests_impl::make_elastic_model();
  const auto chain = depth_series_tests_impl::make_fk_chain();
  const auto &groups = chain.groups;
  const auto &setup = chain.setup;
  const auto &states = chain.states;

  const auto series =
      fk_injection::impl::depth_series(groups, states, model, setup);

  EXPECT_EQ(series.number_of_groups, static_cast<int>(groups.groups.size()));
  EXPECT_EQ(series.coefficient_count, setup.sizes.number_of_stored_frequencies);

  // Every coefficient must be finite.
  double max_s0s1 = 0.0;
  for (int g = 0; g < series.number_of_groups; ++g) {
    for (int s = 0; s < fk_injection::impl::number_of_series; ++s) {
      for (int k = 0; k < series.coefficient_count; ++k) {
        const double coeff = series.coefficients[series.index(g, s, k)];
        EXPECT_TRUE(std::isfinite(coeff))
            << "non-finite coefficient at (g=" << g << ", s=" << s
            << ", k=" << k << "): " << coeff;
        if (s == fk_injection::impl::series_horizontal ||
            s == fk_injection::impl::series_vertical) {
          if (std::fabs(coeff) > max_s0s1)
            max_s0s1 = std::fabs(coeff);
        }
      }
    }
  }
  EXPECT_GT(max_s0s1, 1.0e-6)
      << "max |coefficient| over S0 and S1 slots is too small; displacement "
         "series appears degenerate";
}

// ============================================================================
// NormalizationIndependentOfResampling
//
// With the dt_min inverse-FFT normalization, the reconstructed amplitude is
// independent of NP_RESAMP: the same model/wave at NP=1 (dt=0.1) and NP=2
// (dt=0.05), both with dt_min=0.1, agree at common physical times. The old
// SEM-step normalization made the NP=2 case twice as large.
// ============================================================================
TEST(DepthSeries, NormalizationIndependentOfResampling) {
  const auto model = depth_series_tests_impl::make_elastic_model();
  const auto wave = depth_series_tests_impl::make_wave_p();

  // tau=0 points (x=y=0) at distinct depths.
  const std::vector<double> zs = { 5000.0, 15000.0, 25000.0 };
  const int npts = static_cast<int>(zs.size());
  const auto points = depth_series_tests_impl::make_eval_points(
      std::vector<double>(npts, 0.0), std::vector<double>(npts, 0.0), zs);

  auto make_window = [](double dt, int nstep) {
    fk_injection::TimeWindow w;
    w.dt = static_cast<type_real>(dt);
    w.nstep = nstep;
    w.frequency_max = static_cast<type_real>(1.0);
    w.frequency_sampling = static_cast<type_real>(10.0); // dt_min = 0.1
    w.time_window_length = static_cast<type_real>(40.0);
    return w;
  };
  const auto window_np1 = make_window(0.1, 300);  // NP=1
  const auto window_np2 = make_window(0.05, 600); // NP=2, same dt_min

  const auto result1 =
      fk_injection::solve(model, wave, window_np1, points,
                          fk_injection::field_derivative::displacement);
  const auto result2 =
      fk_injection::solve(model, wave, window_np2, points,
                          fk_injection::field_derivative::displacement);
  ASSERT_EQ(result1.resampling_rate(), 1);
  ASSERT_EQ(result2.resampling_rate(), 2);

  // Row-major (point, component, step).
  const int components = fk_injection::number_of_sample_components;
  std::vector<type_real> out1(static_cast<std::size_t>(npts) * components *
                              300);
  std::vector<type_real> out2(static_cast<std::size_t>(npts) * components *
                              600);
  result1.sample(0, 300, out1);
  result2.sample(0, 600, out2);
  auto h1 = [&](int p, int c, int n) {
    return out1[((p * components + c) * 300) + n];
  };
  auto h2 = [&](int p, int c, int n) {
    return out2[((p * components + c) * 600) + n];
  };

  // SEM step n at NP=1 (time n*0.1) matches SEM step 2n at NP=2 (2n*0.05).
  double peak = 0.0;
  for (int p = 0; p < npts; ++p)
    for (int c = 0; c < 3; ++c)
      for (int n = 0; n < 300; ++n)
        peak = std::max(peak, std::fabs(static_cast<double>(h1(p, c, n))));
  ASSERT_GT(peak, 1.0e-3);

  double max_abs = 0.0;
  for (int p = 0; p < npts; ++p)
    for (int c = 0; c < 3; ++c)
      for (int n = 30; n < 300; ++n) // skip the onset taper zone
        max_abs =
            std::max(max_abs, std::fabs(static_cast<double>(h1(p, c, n)) -
                                        static_cast<double>(h2(p, c, 2 * n))));
  EXPECT_LT(max_abs / peak, 1.0e-5)
      << "NP=1 and NP=2 disagree after the dt_min normalization";
}

namespace depth_series_tests_impl {

/** @brief Two fluid layers (densities 1000, 1100) over crust and half-space. */
fk_injection::LayeredModel make_two_fluid_model() {
  fk_injection::AcousticLayer upper;
  upper.density = 1000;
  upper.p_velocity = 1500;
  upper.thickness = 1000;

  fk_injection::AcousticLayer lower;
  lower.density = 1100;
  lower.p_velocity = 1600;
  lower.thickness = 2000;

  fk_injection::ElasticIsotropicLayer crust;
  crust.density = 2800;
  crust.p_velocity = 6000;
  crust.s_velocity = 3464;
  crust.thickness = 20000;

  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3380;
  halfspace.p_velocity = 8100;
  halfspace.s_velocity = 4500;

  return fk_injection::LayeredModel({ upper, lower }, { crust, halfspace });
}

/** @brief Spectrum of each depth group, row-major (series, frequency). */
std::vector<std::vector<std::complex<double>>>
group_spectra(const fk_injection::LayeredModel &model,
              const fk_injection::impl::DepthGroups &groups,
              const fk_injection::impl::FkSetup &setup) {
  const auto states = fk_injection::impl::solve_layer_stack(model, setup);
  const int fft_samples = setup.sizes.number_of_fft_samples;
  std::vector<std::vector<std::complex<double>>> spectra;
  for (const auto &group : groups.groups) {
    spectra.emplace_back(fk_injection::impl::number_of_series * fft_samples);
    fk_injection::impl::depth_spectrum(group, states, model, setup,
                                       spectra.back().data(), fft_samples);
  }
  return spectra;
}

} // namespace depth_series_tests_impl

// ============================================================================
// AcousticKinematicSeriesIgnoreDerivative
//
// Acoustic Stacey injection consumes the incident displacement and the
// potential rate, so acoustic kinematic series stay displacement when elastic
// ones become velocity.
// ============================================================================
TEST(DepthSeries, AcousticKinematicSeriesIgnoreDerivative) {
  const auto model = depth_series_tests_impl::make_two_fluid_model();
  const auto wave = depth_series_tests_impl::make_wave_p();
  const auto window = depth_series_tests_impl::make_window();
  const double p = wave.ray_parameter(model);

  const auto points = depth_series_tests_impl::make_eval_points(
      { 0.0, 0.0 }, { 0.0, 0.0 }, { 500.0, 10000.0 });
  const auto groups = fk_injection::impl::group_by_depth(
      points, model, wave, p, fk_injection::impl::depth_tolerance(10000.0));
  ASSERT_EQ(groups.groups.size(), 2u);

  const auto displacement =
      fk_injection::impl::prepare(model, wave, window, p, groups,
                                  fk_injection::field_derivative::displacement);
  const auto velocity = fk_injection::impl::prepare(
      model, wave, window, p, groups, fk_injection::field_derivative::velocity);
  const auto displacement_spectra =
      depth_series_tests_impl::group_spectra(model, groups, displacement);
  const auto velocity_spectra =
      depth_series_tests_impl::group_spectra(model, groups, velocity);

  const int fft_samples = displacement.sizes.number_of_fft_samples;
  const int acoustic = groups.point_group[0];
  const int elastic = groups.point_group[1];
  ASSERT_EQ(groups.groups[acoustic].medium, specfem::plugin::Medium::acoustic);
  ASSERT_EQ(groups.groups[elastic].medium, specfem::plugin::Medium::elastic);

  for (const int series : { fk_injection::impl::series_horizontal,
                            fk_injection::impl::series_vertical }) {
    for (int f = 1; f < static_cast<int>(displacement.omega.size()); ++f) {
      const int entry = series * fft_samples + f;
      const auto acoustic_value = displacement_spectra[acoustic][entry];
      EXPECT_LT(std::abs(velocity_spectra[acoustic][entry] - acoustic_value),
                1.0e-9 * (std::abs(acoustic_value) + 1.0e-30))
          << "acoustic series " << series << ", frequency " << f;

      const auto elastic_value = displacement_spectra[elastic][entry];
      const std::complex<double> i_omega(0.0, displacement.omega[f]);
      EXPECT_LT(
          std::abs(velocity_spectra[elastic][entry] - i_omega * elastic_value),
          1.0e-9 * (std::abs(i_omega * elastic_value) + 1.0e-30))
          << "elastic series " << series << ", frequency " << f;
    }
  }
}

// ============================================================================
// HorizontalDensityFollowsLayerResolution
//
// u_x = -i p^2/rho (P/k). SPECFEM3D resolves a point on a layer's bottom
// boundary into the layer below, so rho there is the underlying layer's: the
// next fluid's inside the column, the crust's at the fluid-solid interface.
// ============================================================================
TEST(DepthSeries, HorizontalDensityFollowsLayerResolution) {
  const auto model = depth_series_tests_impl::make_two_fluid_model();
  const auto wave = depth_series_tests_impl::make_wave_p();
  const auto window = depth_series_tests_impl::make_window();
  const double p = wave.ray_parameter(model);

  // Depth, and the density u_x must use there.
  const std::vector<std::pair<double, double>> cases = {
    { 500.0, 1000.0 },  // inside the upper fluid
    { 1000.0, 1100.0 }, // bottom of the upper fluid: lower fluid's
    { 2000.0, 1100.0 }, // inside the lower fluid
    { 3000.0, 2800.0 }, // fluid-solid interface: the crust's
  };

  for (const auto &[depth, density] : cases) {
    auto points =
        depth_series_tests_impl::make_eval_points({ 0.0 }, { 0.0 }, { depth });
    points.medium = { specfem::plugin::Medium::acoustic };
    const auto groups = fk_injection::impl::group_by_depth(
        points, model, wave, p, fk_injection::impl::depth_tolerance(depth));
    ASSERT_EQ(groups.groups.size(), 1u);
    const auto &group = groups.groups[0];

    const auto setup = fk_injection::impl::prepare(
        model, wave, window, p, groups,
        fk_injection::field_derivative::displacement);
    const auto states = fk_injection::impl::solve_layer_stack(model, setup);
    const auto spectra =
        depth_series_tests_impl::group_spectra(model, groups, setup);

    const auto &layer = model.fluid_layers()[group.layer];
    double peak = 0.0;
    for (int f = 1; f < static_cast<int>(setup.omega.size()); ++f) {
      const auto state = fk_injection::impl::acoustic_state_in_layer(
          layer, std::complex<double>(setup.omega[f], 0.0), group.height, p,
          states.acoustic_bottom(f, group.layer));
      const std::complex<double> expected =
          std::complex<double>(0.0, -p * p / density) * state.pressure_over_k *
          setup.apodization[f];
      const auto actual = spectra[0][fk_injection::impl::series_horizontal *
                                         setup.sizes.number_of_fft_samples +
                                     f];
      EXPECT_LT(std::abs(actual - expected),
                1.0e-9 * (std::abs(expected) + 1.0e-30))
          << "depth " << depth << ", frequency " << f;
      peak = std::max(peak, std::abs(expected));
    }
    EXPECT_GT(peak, 0.0) << "depth " << depth;
  }
}
