#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/impl/depth_groups.hpp"
#include "specfem/injection/fk/impl/field_recovery.hpp"
#include "specfem/injection/fk/impl/fk_setup.hpp"
#include "specfem/injection/fk/impl/layer_stack.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Helpers in a named namespace (no anonymous namespaces; unity-build safe).
// ---------------------------------------------------------------------------
namespace layer_stack_tests_impl {

/** @brief Elastic-only model: one finite layer + halfspace. */
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

/** @brief Fluid-over-solid model: fluid {rho=1020,vp=1500,H=2000} + elastic
 *  model below. */
specfem::injection::fk::LayeredModel make_fluid_model() {
  specfem::injection::fk::AcousticLayer fluid;
  fluid.density = static_cast<type_real>(1020.0);
  fluid.p_velocity = static_cast<type_real>(1500.0);
  fluid.thickness = static_cast<type_real>(2000.0);

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

  return specfem::injection::fk::LayeredModel({ fluid },
                                              { finite_layer, halfspace });
}

/** @brief Incident P wave: theta=25 deg, phi=0, unit amplitude. */
specfem::injection::fk::IncidentWave make_wave_p() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::p;
  wave.azimuth_phi = static_cast<type_real>(0.0);
  wave.take_off_theta =
      static_cast<type_real>(25.0 * 3.141592653589793 / 180.0);
  wave.origin_x = static_cast<type_real>(0.0);
  wave.origin_y = static_cast<type_real>(0.0);
  wave.origin_z = static_cast<type_real>(0.0);
  wave.origin_time = static_cast<type_real>(0.0);
  wave.amplitude = static_cast<type_real>(1.0);
  wave.gaussian_half_duration = static_cast<type_real>(0.0);
  return wave;
}

/** @brief Incident SV wave: theta=25 deg, phi=0, unit amplitude. */
specfem::injection::fk::IncidentWave make_wave_sv() {
  specfem::injection::fk::IncidentWave wave;
  wave.type = specfem::injection::fk::incident_wave_type::sv;
  wave.azimuth_phi = static_cast<type_real>(0.0);
  wave.take_off_theta =
      static_cast<type_real>(25.0 * 3.141592653589793 / 180.0);
  wave.origin_x = static_cast<type_real>(0.0);
  wave.origin_y = static_cast<type_real>(0.0);
  wave.origin_z = static_cast<type_real>(0.0);
  wave.origin_time = static_cast<type_real>(0.0);
  wave.amplitude = static_cast<type_real>(1.0);
  wave.gaussian_half_duration = static_cast<type_real>(0.0);
  return wave;
}

/** @brief Standard time window used across all tests. */
specfem::injection::fk::TimeWindow make_window() {
  specfem::injection::fk::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 512;
  window.frequency_max = static_cast<type_real>(1.0);
  window.frequency_sampling = static_cast<type_real>(4.0);
  window.time_window_length = static_cast<type_real>(50.0);
  return window;
}

} // namespace layer_stack_tests_impl

// ============================================================================
// Test 1: HalfspaceTopIsEtimesB
//
// elastic_top(f, n_elastic-1) is the state at the top of the halfspace.  At
// that interface the propagator is the identity (zz=0 => G=I), so the state
// must equal E * b(omega) component-wise.
// ============================================================================

TEST(LayerStack, HalfspaceTopIsEtimesB) {
  const auto model = layer_stack_tests_impl::make_elastic_model();
  const auto wave = layer_stack_tests_impl::make_wave_p();
  const auto window = layer_stack_tests_impl::make_window();

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  specfem::injection::fk_impl::DepthGroups groups; // empty, span 0
  const auto setup = specfem::injection::fk_impl::prepare(
      model, wave, window, p, groups,
      specfem::injection::fk::field_derivative::displacement,
      /*compute_traction=*/true);
  const auto states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);

  const int n_elastic = model.number_of_elastic_layers();
  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  const auto &modes = states.modes_at(f);
  const auto expected = setup.e_mat * modes;
  const auto &actual = states.elastic_top(f, n_elastic - 1);

  constexpr double kTol = 1.0e-9;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(actual[i].real(), expected[i].real(), kTol)
        << "halfspace-top component " << i << " real";
    EXPECT_NEAR(actual[i].imag(), expected[i].imag(), kTol)
        << "halfspace-top component " << i << " imag";
  }
}

// ============================================================================
// Test 2: FreeSurfaceTractionVanishesP
//
// Free surface is the top of elastic layer 0.  Components 2 (sigma_xz) and 3
// (sigma_zz) of the state there must vanish; this is the boundary condition
// imposed by the halfspace solve.
// ============================================================================

TEST(LayerStack, FreeSurfaceTractionVanishesP) {
  const auto model = layer_stack_tests_impl::make_elastic_model();
  const auto wave = layer_stack_tests_impl::make_wave_p();
  const auto window = layer_stack_tests_impl::make_window();

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  specfem::injection::fk_impl::DepthGroups groups;
  const auto setup = specfem::injection::fk_impl::prepare(
      model, wave, window, p, groups,
      specfem::injection::fk::field_derivative::displacement,
      /*compute_traction=*/true);
  const auto states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  // Free-surface state: top of the shallowest elastic layer (layer 0).
  const auto &s = states.elastic_top(f, 0);

  // Displacement scale to normalise traction residuals.
  const double disp_scale = Kokkos::abs(s[0]) + Kokkos::abs(s[1]) + 1.0e-30;
  EXPECT_GT(disp_scale, 1.0e-12) << "displacement scale too small (degenerate)";

  const double rho_hs =
      static_cast<double>(model.elastic_layers().back().density);
  const double vs_hs =
      static_cast<double>(model.elastic_layers().back().s_velocity);
  const double mu = rho_hs * vs_hs * vs_hs;
  const double traction_scale = 2.0 * mu * setup.omega[f] * p * disp_scale;

  EXPECT_LT(Kokkos::abs(s[2]), 1.0e-6 * traction_scale)
      << "sigma_xz at free surface (P) is not small: " << Kokkos::abs(s[2])
      << " vs scale " << traction_scale;
  EXPECT_LT(Kokkos::abs(s[3]), 1.0e-6 * traction_scale)
      << "sigma_zz at free surface (P) is not small: " << Kokkos::abs(s[3])
      << " vs scale " << traction_scale;
}

// ============================================================================
// Test 3: FreeSurfaceTractionVanishesSV
//
// Same as Test 2 but for incident SV.
// ============================================================================

TEST(LayerStack, FreeSurfaceTractionVanishesSV) {
  const auto model = layer_stack_tests_impl::make_elastic_model();
  const auto wave = layer_stack_tests_impl::make_wave_sv();
  const auto window = layer_stack_tests_impl::make_window();

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  specfem::injection::fk_impl::DepthGroups groups;
  const auto setup = specfem::injection::fk_impl::prepare(
      model, wave, window, p, groups,
      specfem::injection::fk::field_derivative::displacement,
      /*compute_traction=*/true);
  const auto states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  const auto &s = states.elastic_top(f, 0);

  const double disp_scale = Kokkos::abs(s[0]) + Kokkos::abs(s[1]) + 1.0e-30;
  EXPECT_GT(disp_scale, 1.0e-12) << "displacement scale too small (degenerate)";

  const double rho_hs =
      static_cast<double>(model.elastic_layers().back().density);
  const double vs_hs =
      static_cast<double>(model.elastic_layers().back().s_velocity);
  const double mu = rho_hs * vs_hs * vs_hs;
  const double traction_scale = 2.0 * mu * setup.omega[f] * p * disp_scale;

  EXPECT_LT(Kokkos::abs(s[2]), 1.0e-6 * traction_scale)
      << "sigma_xz at free surface (SV) is not small: " << Kokkos::abs(s[2])
      << " vs scale " << traction_scale;
  EXPECT_LT(Kokkos::abs(s[3]), 1.0e-6 * traction_scale)
      << "sigma_zz at free surface (SV) is not small: " << Kokkos::abs(s[3])
      << " vs scale " << traction_scale;
}

// ============================================================================
// Test 4: FluidFreeSurfacePressureSmall
//
// Fluid-over-solid model.  The acoustic free surface is the top of fluid
// layer 0.  Propagate the acoustic state from the bottom of the fluid layer
// to its top; the free-surface BC requires pressure_over_k == 0 there.
// ============================================================================

TEST(LayerStack, FluidFreeSurfacePressureSmall) {
  const auto model = layer_stack_tests_impl::make_fluid_model();
  const auto wave = layer_stack_tests_impl::make_wave_p();
  const auto window = layer_stack_tests_impl::make_window();

  const double p = specfem::injection::fk_impl::ray_parameter(model, wave);
  specfem::injection::fk_impl::DepthGroups groups;
  const auto setup = specfem::injection::fk_impl::prepare(
      model, wave, window, p, groups,
      specfem::injection::fk::field_derivative::displacement,
      /*compute_traction=*/true);
  const auto states =
      specfem::injection::fk_impl::solve_layer_stack(model, setup);

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  // Propagate from the bottom of fluid layer 0 to its top.
  const auto &fluid0 = model.fluid_layers()[0];
  const auto top = specfem::injection::fk_impl::acoustic_state_in_layer(
      fluid0, Kokkos::complex<double>(setup.omega[f], 0.0),
      static_cast<type_real>(fluid0.thickness), static_cast<type_real>(p),
      states.acoustic_bottom(f, 0));

  const double scale =
      Kokkos::abs(states.acoustic_bottom(f, 0).u_z) +
      Kokkos::abs(states.acoustic_bottom(f, 0).pressure_over_k) + 1.0e-30;

  EXPECT_LT(Kokkos::abs(top.pressure_over_k), 1.0e-5 * scale)
      << "pressure_over_k at acoustic free surface is not small: "
      << Kokkos::abs(top.pressure_over_k) << " vs scale " << scale;
}
