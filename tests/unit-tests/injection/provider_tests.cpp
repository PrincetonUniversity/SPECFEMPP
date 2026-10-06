#include "specfem/injection/fk/solver.hpp"
#include "specfem/injection/fk_provider.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"
#include "specfem/utilities/cubic_bspline.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: build an EvalPoints with n points at the given z-coords.
// Normal vectors point in +z direction; Lamé ratios set to representative
// elastic values.
// ---------------------------------------------------------------------------
static specfem::injection::fk::EvalPoints
make_eval_points_with_normals(const std::vector<double> &z_coords) {
  const int n = static_cast<int>(z_coords.size());

  Kokkos::View<type_real *, Kokkos::HostSpace> hx("hx", n), hy("hy", n),
      hz("hz", n);
  Kokkos::View<type_real *, Kokkos::HostSpace> hnx("hnx", n), hny("hny", n),
      hnz("hnz", n);
  Kokkos::View<type_real *, Kokkos::HostSpace> hxi1("hxi1", n), hxim("hxim", n),
      hbulk("hbulk", n);

  for (int i = 0; i < n; ++i) {
    hx(i) = static_cast<type_real>(0);
    hy(i) = static_cast<type_real>(0);
    hz(i) = static_cast<type_real>(z_coords[i]);
    hnx(i) = static_cast<type_real>(0);
    hny(i) = static_cast<type_real>(0);
    hnz(i) = static_cast<type_real>(1);
    hxi1(i) = static_cast<type_real>(2.0); // xi1 = (lambda + 2*mu) / mu
    hxim(i) = static_cast<type_real>(1.0); // xim = mu / (2*mu)
    hbulk(i) = static_cast<type_real>(0.333);
  }

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *>("x", n);
  ep.y = Kokkos::View<type_real *>("y", n);
  ep.z = Kokkos::View<type_real *>("z", n);
  ep.normal_x = Kokkos::View<type_real *>("normal_x", n);
  ep.normal_y = Kokkos::View<type_real *>("normal_y", n);
  ep.normal_z = Kokkos::View<type_real *>("normal_z", n);
  ep.lame_ratio_xi1 = Kokkos::View<type_real *>("lame_ratio_xi1", n);
  ep.lame_ratio_xim = Kokkos::View<type_real *>("lame_ratio_xim", n);
  ep.lame_ratio_bulk = Kokkos::View<type_real *>("lame_ratio_bulk", n);

  Kokkos::deep_copy(ep.x, hx);
  Kokkos::deep_copy(ep.y, hy);
  Kokkos::deep_copy(ep.z, hz);
  Kokkos::deep_copy(ep.normal_x, hnx);
  Kokkos::deep_copy(ep.normal_y, hny);
  Kokkos::deep_copy(ep.normal_z, hnz);
  Kokkos::deep_copy(ep.lame_ratio_xi1, hxi1);
  Kokkos::deep_copy(ep.lame_ratio_xim, hxim);
  Kokkos::deep_copy(ep.lame_ratio_bulk, hbulk);

  return ep;
}

// ---------------------------------------------------------------------------
// Helper: build the standard small elastic crust+mantle model used in several
// tests below.
// ---------------------------------------------------------------------------
static specfem::injection::fk::LayeredModel make_crust_mantle_model() {
  specfem::injection::fk::ElasticIsotropicLayer crust;
  crust.p_velocity = static_cast<type_real>(6000);
  crust.s_velocity = static_cast<type_real>(3500);
  crust.density = static_cast<type_real>(2800);
  crust.thickness = static_cast<type_real>(35000);

  specfem::injection::fk::ElasticIsotropicLayer mantle;
  mantle.p_velocity = static_cast<type_real>(8100);
  mantle.s_velocity = static_cast<type_real>(4500);
  mantle.density = static_cast<type_real>(3300);
  mantle.thickness = static_cast<type_real>(0); // half-space

  return specfem::injection::fk::LayeredModel({}, { crust, mantle });
}

// ---------------------------------------------------------------------------
// Helper: build the standard incident P wave used in these tests.
// ---------------------------------------------------------------------------
static specfem::injection::fk::IncidentWave make_incident_p_wave() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0);
  wave.take_off_theta = static_cast<type_real>(20.0 * 3.14159265358979 / 180.0);
  wave.origin_x = static_cast<type_real>(0);
  wave.origin_y = static_cast<type_real>(0);
  wave.origin_z = static_cast<type_real>(0);
  wave.origin_time = static_cast<type_real>(5);
  wave.amplitude = static_cast<type_real>(1);
  wave.gaussian_half_duration = static_cast<type_real>(2);
  return wave;
}

// ---------------------------------------------------------------------------
// Helper: build the small time window used in these tests.
// ---------------------------------------------------------------------------
static specfem::injection::fk::TimeWindow make_small_time_window() {
  specfem::injection::fk::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 256;
  window.frequency_max = static_cast<type_real>(1);
  window.frequency_sampling = static_cast<type_real>(2);
  window.time_window_length = static_cast<type_real>(40);
  return window;
}

// ===========================================================================
// Test 1: FkProviderReconstructsFinite
//
// Build a small elastic model (crust + mantle), incident P, small time window,
// a few evaluation points.  After initialize() + ensure_window():
//   - buffer shape is (npoints, 6, nstep)
//   - every value in the buffer is finite
//   - displacement components are not all-zero
// ===========================================================================
TEST(FkProvider, FkProviderReconstructsFinite) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  const auto points =
      make_eval_points_with_normals({ 0, 5000, 15000, 30000, 40000 });
  const int npoints = points.size();
  const int nstep = window.nstep;

  specfem::injection::fk_provider<specfem::element::dimension_tag::dim3>
      provider(model, wave, window, points, /*compute_traction=*/true);

  provider.initialize();
  provider.ensure_window(nstep - 1);

  const specfem::injection::InjectionFrameBuffer &buf = provider.frame_buffer();

  // Shape checks
  EXPECT_EQ(buf.number_of_points(), npoints);
  EXPECT_EQ(buf.number_of_components(), 6); // disp (3) + traction (3)
  EXPECT_EQ(buf.number_of_steps(), nstep);

  // Copy to host for value checks.
  auto h_vals = Kokkos::create_mirror_view(buf.values());
  Kokkos::deep_copy(h_vals, buf.values());

  bool all_finite = true;
  double max_abs_disp = 0.0;

  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < 6; ++ic) {
      for (int is = 0; is < nstep; ++is) {
        const double v = static_cast<double>(h_vals(ip, ic, is));
        if (!std::isfinite(v)) {
          all_finite = false;
        }
        if (ic < 3 && std::fabs(v) > max_abs_disp) {
          max_abs_disp = std::fabs(v);
        }
      }
    }
  }

  EXPECT_TRUE(all_finite) << "buffer contains NaN or Inf";
  EXPECT_GT(max_abs_disp, 1.0e-30) << "displacement components are all-zero";
}

// ===========================================================================
// Test 2: FkProviderEnsureWindowIsIdempotent
//
// Calling ensure_window() a second time must not crash and must leave the
// buffer values unchanged (the no-op guard works).
// ===========================================================================
TEST(FkProvider, FkProviderEnsureWindowIsIdempotent) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  const auto points = make_eval_points_with_normals({ 0, 10000 });

  specfem::injection::fk_provider<specfem::element::dimension_tag::dim3>
      provider(model, wave, window, points, /*compute_traction=*/false);

  provider.initialize();
  provider.ensure_window(0);

  // Snapshot value at (point=0, comp=0, step=10)
  auto h_vals_1 = Kokkos::create_mirror_view(provider.frame_buffer().values());
  Kokkos::deep_copy(h_vals_1, provider.frame_buffer().values());
  const double v_before = static_cast<double>(h_vals_1(0, 0, 10));

  // Second call — must be a no-op
  provider.ensure_window(5);

  auto h_vals_2 = Kokkos::create_mirror_view(provider.frame_buffer().values());
  Kokkos::deep_copy(h_vals_2, provider.frame_buffer().values());
  const double v_after = static_cast<double>(h_vals_2(0, 0, 10));

  EXPECT_DOUBLE_EQ(v_before, v_after);
}

// ===========================================================================
// Test 3: FkProviderConsistency
//
// At a SEM time step that is an exact multiple of NP_RESAMP (the resampling
// rate), say step = k * NP_RESAMP, the B-spline abscissa is the integer k.
// The reconstructed value load_on_device(k*NP_RESAMP, point, comp) must equal
// cubic_bspline::evaluate(coeffs_double, ncoef, k) within a float tolerance.
//
// This verifies the abscissa mapping  abscissa = istep / NP_RESAMP.
// ===========================================================================
TEST(FkProvider, FkProviderConsistency) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  // Single point at the surface for simplicity.
  const auto points = make_eval_points_with_normals({ 0 });

  specfem::injection::fk_provider<specfem::element::dimension_tag::dim3>
      provider(model, wave, window, points, /*compute_traction=*/false);

  provider.initialize();
  provider.ensure_window(window.nstep - 1);

  // Retrieve raw FK result to read coefficient data.
  // We run the FK solve independently (same inputs) to get the resampling rate
  // and coefficient array for comparison.
  const auto result = specfem::injection::fk::solve(model, wave, window, points,
                                                    /*compute_traction=*/false);

  const int np_resamp = result.resampling_rate();
  const int ncoef = result.coefficient_count();
  ASSERT_GT(np_resamp, 0) << "resampling rate must be positive";
  ASSERT_GT(ncoef, 0) << "coefficient count must be positive";

  // Choose k = 2 (abscissa = 2) so we are well inside the coefficient range.
  const int k = 2;
  const int sem_step = k * np_resamp;
  ASSERT_LT(sem_step, window.nstep)
      << "chosen step out of range for this window";

  // Displacement component 0, point 0.
  const int ipoint = 0;
  const int icomp = 0;

  // Build double coefficient array from the FkResult.
  auto h_disp = Kokkos::create_mirror_view(result.displacement());
  Kokkos::deep_copy(h_disp, result.displacement());
  std::vector<double> coeff_d(ncoef);
  for (int i = 0; i < ncoef; ++i) {
    coeff_d[i] = static_cast<double>(h_disp(ipoint, icomp, i));
  }

  const double expected = specfem::utilities::cubic_bspline::evaluate(
      coeff_d.data(), ncoef, static_cast<double>(k));

  // Read the reconstructed value from the buffer.
  auto h_buf = Kokkos::create_mirror_view(provider.frame_buffer().values());
  Kokkos::deep_copy(h_buf, provider.frame_buffer().values());
  const double actual = static_cast<double>(h_buf(ipoint, icomp, sem_step));

  // Float precision: allow 1e-5 relative tolerance.
  EXPECT_NEAR(actual, expected, 1.0e-5 * std::max(std::fabs(expected), 1.0e-30))
      << "buffer value at exact B-spline knot does not match direct evaluate()";
}
