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

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *, Kokkos::HostSpace>("x", n);
  ep.y = Kokkos::View<type_real *, Kokkos::HostSpace>("y", n);
  ep.z = Kokkos::View<type_real *, Kokkos::HostSpace>("z", n);
  ep.normal_x = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_x", n);
  ep.normal_y = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_y", n);
  ep.normal_z = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_z", n);
  ep.lame_ratio_xi1 =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_xi1", n);
  ep.lame_ratio_xim =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_xim", n);
  ep.lame_ratio_bulk =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_bulk", n);

  for (int i = 0; i < n; ++i) {
    ep.x(i) = static_cast<type_real>(0);
    ep.y(i) = static_cast<type_real>(0);
    ep.z(i) = static_cast<type_real>(z_coords[i]);
    ep.normal_x(i) = static_cast<type_real>(0);
    ep.normal_y(i) = static_cast<type_real>(0);
    ep.normal_z(i) = static_cast<type_real>(1);
    ep.lame_ratio_xi1(i) = static_cast<type_real>(2.0); // (lambda + 2*mu) / mu
    ep.lame_ratio_xim(i) = static_cast<type_real>(1.0); // mu / (2*mu)
    ep.lame_ratio_bulk(i) = static_cast<type_real>(0.333);
  }

  return ep;
}

// ---------------------------------------------------------------------------
// Helper: build an EvalPoints at the given (x, z) coordinates with normals and
// Lamé ratios.  A non-zero x gives a non-zero FK time delay tau.
// ---------------------------------------------------------------------------
static specfem::injection::fk::EvalPoints
make_eval_points_xz(const std::vector<double> &xs,
                    const std::vector<double> &zs) {
  const int n = static_cast<int>(xs.size());

  specfem::injection::fk::EvalPoints ep;
  ep.x = Kokkos::View<type_real *, Kokkos::HostSpace>("x", n);
  ep.y = Kokkos::View<type_real *, Kokkos::HostSpace>("y", n);
  ep.z = Kokkos::View<type_real *, Kokkos::HostSpace>("z", n);
  ep.normal_x = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_x", n);
  ep.normal_y = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_y", n);
  ep.normal_z = Kokkos::View<type_real *, Kokkos::HostSpace>("normal_z", n);
  ep.lame_ratio_xi1 =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_xi1", n);
  ep.lame_ratio_xim =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_xim", n);
  ep.lame_ratio_bulk =
      Kokkos::View<type_real *, Kokkos::HostSpace>("lame_ratio_bulk", n);

  for (int i = 0; i < n; ++i) {
    ep.x(i) = static_cast<type_real>(xs[i]);
    ep.y(i) = static_cast<type_real>(0);
    ep.z(i) = static_cast<type_real>(zs[i]);
    ep.normal_x(i) = static_cast<type_real>(0);
    ep.normal_y(i) = static_cast<type_real>(0);
    ep.normal_z(i) = static_cast<type_real>(1);
    ep.lame_ratio_xi1(i) = static_cast<type_real>(2.0);
    ep.lame_ratio_xim(i) = static_cast<type_real>(1.0);
    ep.lame_ratio_bulk(i) = static_cast<type_real>(0.333);
  }

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

  // Shape checks: the buffer holds a window of W <= nstep steps.
  const int W = buf.number_of_steps();
  EXPECT_EQ(buf.number_of_points(), npoints);
  EXPECT_EQ(buf.number_of_components(), 6); // disp (3) + traction (3)
  EXPECT_GT(W, 0);
  EXPECT_LE(W, nstep);

  // Copy to host for value checks.
  auto h_vals = Kokkos::create_mirror_view(buf.values());
  Kokkos::deep_copy(h_vals, buf.values());

  bool all_finite = true;
  double max_abs_disp = 0.0;

  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < 6; ++ic) {
      for (int is = 0; is < W; ++is) {
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
// Test 3: LoadMatchesFkResultSample
//
// load_on_device maps an absolute SEM step to the windowed buffer index
// istep - window_start.  With a small window (so the window start is non-zero)
// and a point with tau != 0 (lateral offset) and NP > 1, the values read back
// through load_on_device must match an independent full FkResult::sample at the
// same absolute steps.
// ===========================================================================
TEST(FkProvider, LoadMatchesFkResultSample) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  // Point 0 at the origin (tau = 0); point 1 laterally offset (tau != 0).
  const auto points = make_eval_points_xz({ 0.0, 30000.0 }, { 5000.0, 5000.0 });
  const int npoints = points.size();
  const int nstep = window.nstep;

  specfem::injection::fk_provider<specfem::element::dimension_tag::dim3>
      provider(model, wave, window, points, /*compute_traction=*/false,
               specfem::injection::fk::field_derivative::displacement,
               /*window_steps=*/32);

  provider.initialize();
  const int W = provider.frame_buffer().number_of_steps();
  ASSERT_LT(W, nstep) << "test needs a window smaller than the full series";

  // Request a step beyond the first window so the window start is non-zero.
  const int probe = W + 3;
  provider.ensure_window(probe);
  const int start = (probe / W) * W;
  const int count = std::min(W, nstep - start);
  ASSERT_GT(start, 0);

  // Independent full sample on the same inputs.
  const auto result = specfem::injection::fk::solve(model, wave, window, points,
                                                    /*compute_traction=*/false);
  ASSERT_GT(result.resampling_rate(), 0);
  const int ncomp = provider.frame_buffer().number_of_components();
  Kokkos::View<type_real ***> full("full", npoints, ncomp, nstep);
  result.sample(0, nstep, full);
  auto h_full = Kokkos::create_mirror_view(full);
  Kokkos::deep_copy(h_full, full);

  // Read the window back through the device accessor load_on_device.
  const specfem::injection::InjectionFrameBuffer buffer =
      provider.frame_buffer();
  Kokkos::View<type_real ***> loaded("loaded", npoints, ncomp, count);
  Kokkos::parallel_for(
      "load_window",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 },
                                             { npoints, ncomp, count }),
      KOKKOS_LAMBDA(int ip, int ic, int k) {
        loaded(ip, ic, k) = buffer.load_on_device(start + k, ip, ic);
      });
  auto h_loaded = Kokkos::create_mirror_view(loaded);
  Kokkos::deep_copy(h_loaded, loaded);

  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < ncomp; ++ic) {
      for (int k = 0; k < count; ++k) {
        EXPECT_DOUBLE_EQ(static_cast<double>(h_loaded(ip, ic, k)),
                         static_cast<double>(h_full(ip, ic, start + k)))
            << "mismatch at point=" << ip << " comp=" << ic
            << " step=" << start + k;
      }
    }
  }
}

// ===========================================================================
// Test 4: WindowRolloverMatchesFreshProvider
//
// Rolling a provider forward (ensure_window(0) then ensure_window(W+3)) yields
// the same buffer contents as a fresh provider asked only for W+3, because
// aligned window starts make the result independent of call order.
// ===========================================================================
TEST(FkProvider, WindowRolloverMatchesFreshProvider) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  const auto points = make_eval_points_xz({ 0.0, 30000.0 }, { 5000.0, 5000.0 });

  auto make_provider = [&]() {
    return specfem::injection::fk_provider<
        specfem::element::dimension_tag::dim3>(
        model, wave, window, points, /*compute_traction=*/false,
        specfem::injection::fk::field_derivative::displacement,
        /*window_steps=*/32);
  };

  auto rolled = make_provider();
  rolled.initialize();
  const int W = rolled.frame_buffer().number_of_steps();
  const int probe = W + 3;
  rolled.ensure_window(0);
  rolled.ensure_window(probe);

  auto fresh = make_provider();
  fresh.initialize();
  fresh.ensure_window(probe);

  EXPECT_EQ(rolled.frame_buffer().window_start(),
            fresh.frame_buffer().window_start());

  auto h_rolled = Kokkos::create_mirror_view(rolled.frame_buffer().values());
  Kokkos::deep_copy(h_rolled, rolled.frame_buffer().values());
  auto h_fresh = Kokkos::create_mirror_view(fresh.frame_buffer().values());
  Kokkos::deep_copy(h_fresh, fresh.frame_buffer().values());

  const int npoints = points.size();
  const int ncomp = rolled.frame_buffer().number_of_components();
  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < ncomp; ++ic) {
      for (int k = 0; k < W; ++k) {
        EXPECT_DOUBLE_EQ(static_cast<double>(h_rolled(ip, ic, k)),
                         static_cast<double>(h_fresh(ip, ic, k)))
            << "mismatch at point=" << ip << " comp=" << ic << " k=" << k;
      }
    }
  }
}

// ===========================================================================
// Test 5: SmallWindowStepsHonored
//
// An explicit window_steps request sets the buffer extent exactly, and its
// values match a full independent sample over the same leading steps.
// ===========================================================================
TEST(FkProvider, SmallWindowStepsHonored) {
  const auto model = make_crust_mantle_model();
  const auto wave = make_incident_p_wave();
  const auto window = make_small_time_window();
  const auto points = make_eval_points_with_normals({ 0, 5000 });
  const int npoints = points.size();

  specfem::injection::fk_provider<specfem::element::dimension_tag::dim3>
      provider(model, wave, window, points, /*compute_traction=*/false,
               specfem::injection::fk::field_derivative::displacement,
               /*window_steps=*/7);

  provider.initialize();
  EXPECT_EQ(provider.frame_buffer().number_of_steps(), 7);

  provider.ensure_window(0);

  const auto result = specfem::injection::fk::solve(model, wave, window, points,
                                                    /*compute_traction=*/false);
  const int ncomp = provider.frame_buffer().number_of_components();
  Kokkos::View<type_real ***> full("full", npoints, ncomp, window.nstep);
  result.sample(0, window.nstep, full);
  auto h_full = Kokkos::create_mirror_view(full);
  Kokkos::deep_copy(h_full, full);

  auto h_buf = Kokkos::create_mirror_view(provider.frame_buffer().values());
  Kokkos::deep_copy(h_buf, provider.frame_buffer().values());

  for (int ip = 0; ip < npoints; ++ip) {
    for (int ic = 0; ic < ncomp; ++ic) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_DOUBLE_EQ(static_cast<double>(h_buf(ip, ic, k)),
                         static_cast<double>(h_full(ip, ic, k)))
            << "mismatch at point=" << ip << " comp=" << ic << " k=" << k;
      }
    }
  }
}
