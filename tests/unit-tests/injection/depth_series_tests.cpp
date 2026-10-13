#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/impl/depth_groups.hpp"
#include "specfem/injection/fk/impl/depth_series.hpp"
#include "specfem/injection/fk/impl/driver.hpp"
#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_setup.hpp"
#include "specfem/injection/fk/impl/layer_stack.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers in a named namespace (no anonymous namespaces; unity-build safe).
// ---------------------------------------------------------------------------
namespace depth_series_tests_impl {

/** @brief Build an EvalPoints device view from parallel host arrays. */
specfem::injection::fk::EvalPoints
make_eval_points(const std::vector<double> &xs, const std::vector<double> &ys,
                 const std::vector<double> &zs) {
  const int n = static_cast<int>(xs.size());

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *, Kokkos::HostSpace>("x", n);
  ep.y = Kokkos::View<type_real *, Kokkos::HostSpace>("y", n);
  ep.z = Kokkos::View<type_real *, Kokkos::HostSpace>("z", n);
  for (int i = 0; i < n; ++i) {
    ep.x(i) = static_cast<type_real>(xs[i]);
    ep.y(i) = static_cast<type_real>(ys[i]);
    ep.z(i) = static_cast<type_real>(zs[i]);
  }

  return ep;
}

/** @brief Elastic-only model: one finite layer (H=20000) + halfspace. */
specfem::injection::fk::LayeredModel make_elastic_model() {
  specfem::injection::fk::ElasticIsotropicLayer finite_layer;
  finite_layer.density = static_cast<type_real>(2800.0);
  finite_layer.p_velocity = static_cast<type_real>(6000.0);
  finite_layer.s_velocity = static_cast<type_real>(3464.0);
  finite_layer.thickness = static_cast<type_real>(20000.0);

  specfem::injection::fk::ElasticIsotropicLayer halfspace;
  halfspace.density = static_cast<type_real>(3380.0);
  halfspace.p_velocity = static_cast<type_real>(8100.0);
  halfspace.s_velocity = static_cast<type_real>(4500.0);
  halfspace.thickness = static_cast<type_real>(0.0);

  return specfem::injection::fk::LayeredModel({}, { finite_layer, halfspace });
}

/** @brief Incident P wave: theta=25 deg, phi=0, origin_time=-5.05 s. */
specfem::injection::fk::IncidentWave make_wave_p() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::p;
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
specfem::injection::fk::TimeWindow make_window() {
  specfem::injection::fk::TimeWindow window;
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
  specfem::injection::fk_impl::DepthGroups groups;
  specfem::injection::fk_impl::FkSetup setup;
  specfem::injection::fk_impl::InterfaceStates states;
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

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  const double eps = specfem::injection::fk_impl::depth_tolerance(25000.0);
  const auto groups =
      specfem::injection::fk_impl::group_by_depth(points, model, wave, p, eps);

  const auto setup = specfem::injection::fk_impl::prepare(
      model, wave, window, p, groups,
      specfem::injection::fk::field_derivative::displacement,
      /*compute_traction=*/true);

  const auto states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);

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

  const specfem::injection::fk_impl::DepthGroup &group =
      groups.groups[target_group];

  const int nf_total = setup.sizes.number_of_fft_samples;
  const int nf_stored = setup.sizes.number_of_stored_frequencies;

  std::vector<Kokkos::complex<double>> spec(
      specfem::injection::fk_impl::number_of_series * nf_total,
      Kokkos::complex<double>(0.0, 0.0));

  specfem::injection::fk_impl::depth_spectrum(group, states, model, setup,
                                              spec.data(), nf_total);

  // Check that the group is non-trivial: max |S4| over positive frequencies
  // should exceed 1e-6.
  double max_s4 = 0.0;
  for (int f = 1; f <= nf_stored; ++f) {
    const double s4_abs = Kokkos::abs(
        spec[specfem::injection::fk_impl::series_dx * nf_total + f]);
    if (s4_abs > max_s4)
      max_s4 = s4_abs;
  }
  EXPECT_GT(max_s4, 1.0e-6)
      << "S4 is uniformly tiny; the group appears degenerate";

  // Identity: i * omega * p * S0 == S4. The operators carry the ray parameter
  // as type_real (single precision, matching the reference), so use the same
  // float-rounded value here; otherwise the comparison is dominated by the
  // float-vs-double representation of p (~1e-7), not the identity.
  const double p_val = static_cast<double>(static_cast<type_real>(setup.p));
  for (int f = 1; f <= nf_stored; ++f) {
    const Kokkos::complex<double> s0 =
        spec[specfem::injection::fk_impl::series_horizontal * nf_total + f];
    const Kokkos::complex<double> s4 =
        spec[specfem::injection::fk_impl::series_dx * nf_total + f];

    const Kokkos::complex<double> lhs =
        Kokkos::complex<double>(0.0, setup.omega[f] * p_val) * s0;

    const double abs_diff = Kokkos::abs(lhs - s4);
    const double scale = Kokkos::abs(s4) + 1.0e-30;

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
      specfem::injection::fk_impl::depth_series(groups, states, model, setup);

  EXPECT_EQ(series.number_of_groups, static_cast<int>(groups.groups.size()));
  EXPECT_EQ(series.coefficient_count, setup.sizes.number_of_stored_frequencies);
  EXPECT_EQ(series.number_of_series,
            specfem::injection::fk_impl::number_of_series);

  // Every coefficient must be finite.
  double max_s0s1 = 0.0;
  for (int g = 0; g < series.number_of_groups; ++g) {
    for (int s = 0; s < series.number_of_series; ++s) {
      for (int k = 0; k < series.coefficient_count; ++k) {
        const double coeff = series.coefficients(g, s, k);
        EXPECT_TRUE(std::isfinite(coeff))
            << "non-finite coefficient at (g=" << g << ", s=" << s
            << ", k=" << k << "): " << coeff;
        if (s == specfem::injection::fk_impl::series_horizontal ||
            s == specfem::injection::fk_impl::series_vertical) {
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
// NormalizationIndependentOfResampling (finding #4 / 4b)
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
    specfem::injection::fk::TimeWindow w;
    w.dt = static_cast<type_real>(dt);
    w.nstep = nstep;
    w.frequency_max = static_cast<type_real>(1.0);
    w.frequency_sampling = static_cast<type_real>(10.0); // dt_min = 0.1
    w.time_window_length = static_cast<type_real>(40.0);
    return w;
  };
  const auto window_np1 = make_window(0.1, 300);  // NP=1
  const auto window_np2 = make_window(0.05, 600); // NP=2, same dt_min

  const auto result1 = specfem::injection::fk_impl::run_fk_driver(
      model, wave, window_np1, points, /*traction=*/false,
      specfem::injection::fk::field_derivative::displacement);
  const auto result2 = specfem::injection::fk_impl::run_fk_driver(
      model, wave, window_np2, points, /*traction=*/false,
      specfem::injection::fk::field_derivative::displacement);
  ASSERT_EQ(result1.resampling_rate(), 1);
  ASSERT_EQ(result2.resampling_rate(), 2);

  Kokkos::View<type_real ***> out1("out1", npts, 3, 300);
  Kokkos::View<type_real ***> out2("out2", npts, 3, 600);
  result1.sample(0, 300, out1);
  result2.sample(0, 600, out2);
  auto h1 = Kokkos::create_mirror_view(out1);
  auto h2 = Kokkos::create_mirror_view(out2);
  Kokkos::deep_copy(h1, out1);
  Kokkos::deep_copy(h2, out2);

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
