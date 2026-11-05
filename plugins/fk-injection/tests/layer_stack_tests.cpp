#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/field_recovery.hpp"
#include "impl/layer_stack.hpp"
#include "plane_wave_response.hpp"
#include <cmath>
#include <gtest/gtest.h>

using fk_injection::type_real;

// ---------------------------------------------------------------------------
// Helpers in a named namespace (no anonymous namespaces; unity-build safe).
// ---------------------------------------------------------------------------
namespace layer_stack_tests_impl {

/** @brief Elastic-only model: one finite layer + halfspace. */
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

  return fk_injection::LayeredModel({ finite_layer, halfspace });
}

/** @brief Fluid-over-solid model: fluid {rho=1020,vp=1500,H=2000} + elastic
 *  model below. */
fk_injection::LayeredModel make_fluid_model() {
  fk_injection::AcousticLayer fluid;
  fluid.density = static_cast<type_real>(1020.0);
  fluid.p_velocity = static_cast<type_real>(1500.0);
  fluid.thickness = static_cast<type_real>(2000.0);

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

  return fk_injection::LayeredModel({ fluid, finite_layer, halfspace });
}

/** @brief Incident P wave: theta=25 deg, phi=0, unit amplitude. */
fk_injection::IncidentWave make_wave_p() {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::p;
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
fk_injection::IncidentWave make_wave_sv() {
  fk_injection::IncidentWave wave;
  wave.type = fk_injection::incident_wave_type::sv;
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
fk_injection::TimeWindow make_window() {
  fk_injection::TimeWindow window;
  window.dt = static_cast<type_real>(0.05);
  window.nstep = 512;
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

  const fk_injection::PlaneWaveResponse response(model, wave, window, {});
  const auto &setup = response.setup();
  const auto &states = response.states();

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

  const double p = wave.ray_parameter(model);
  const fk_injection::PlaneWaveResponse response(model, wave, window, {});
  const auto &setup = response.setup();
  const auto &states = response.states();

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  // Free-surface state: top of the shallowest elastic layer (layer 0).
  const auto &s = states.elastic_top(f, 0);

  // Displacement scale to normalise traction residuals.
  const double disp_scale = std::abs(s[0]) + std::abs(s[1]) + 1.0e-30;
  EXPECT_GT(disp_scale, 1.0e-12) << "displacement scale too small (degenerate)";

  const double rho_hs =
      static_cast<double>(model.elastic_layers().back().density);
  const double vs_hs =
      static_cast<double>(model.elastic_layers().back().s_velocity);
  const double mu = rho_hs * vs_hs * vs_hs;
  const double traction_scale = 2.0 * mu * setup.omega[f] * p * disp_scale;

  EXPECT_LT(std::abs(s[2]), 1.0e-6 * traction_scale)
      << "sigma_xz at free surface (P) is not small: " << std::abs(s[2])
      << " vs scale " << traction_scale;
  EXPECT_LT(std::abs(s[3]), 1.0e-6 * traction_scale)
      << "sigma_zz at free surface (P) is not small: " << std::abs(s[3])
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

  const double p = wave.ray_parameter(model);
  const fk_injection::PlaneWaveResponse response(model, wave, window, {});
  const auto &setup = response.setup();
  const auto &states = response.states();

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  const auto &s = states.elastic_top(f, 0);

  const double disp_scale = std::abs(s[0]) + std::abs(s[1]) + 1.0e-30;
  EXPECT_GT(disp_scale, 1.0e-12) << "displacement scale too small (degenerate)";

  const double rho_hs =
      static_cast<double>(model.elastic_layers().back().density);
  const double vs_hs =
      static_cast<double>(model.elastic_layers().back().s_velocity);
  const double mu = rho_hs * vs_hs * vs_hs;
  const double traction_scale = 2.0 * mu * setup.omega[f] * p * disp_scale;

  EXPECT_LT(std::abs(s[2]), 1.0e-6 * traction_scale)
      << "sigma_xz at free surface (SV) is not small: " << std::abs(s[2])
      << " vs scale " << traction_scale;
  EXPECT_LT(std::abs(s[3]), 1.0e-6 * traction_scale)
      << "sigma_zz at free surface (SV) is not small: " << std::abs(s[3])
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

  const double p = wave.ray_parameter(model);
  const fk_injection::PlaneWaveResponse response(model, wave, window, {});
  const auto &setup = response.setup();
  const auto &states = response.states();

  const int f = 10;
  ASSERT_LT(f, static_cast<int>(setup.omega.size()));

  // Propagate from the bottom of fluid layer 0 to its top.
  const auto &fluid0 = model.fluid_layers()[0];
  const auto top = fk_injection::impl::acoustic_state_in_layer(
      fluid0, std::complex<double>(setup.omega[f], 0.0),
      static_cast<double>(fluid0.thickness), p, states.acoustic_bottom(f, 0));

  const double scale = std::abs(states.acoustic_bottom(f, 0).u_z) +
                       std::abs(states.acoustic_bottom(f, 0).pressure_over_k) +
                       1.0e-30;

  EXPECT_LT(std::abs(top.pressure_over_k), 1.0e-5 * scale)
      << "pressure_over_k at acoustic free surface is not small: "
      << std::abs(top.pressure_over_k) << " vs scale " << scale;
}

// 30 km of water over a solid, P at 20 deg take-off: |R| = 0.6817 and a 39.41 s
// round trip, from the fluid-solid boundary conditions solved independently.
TEST(FluidReverberation, DecayTimeFromBottomReflection) {
  fk_injection::AcousticLayer water;
  water.density = static_cast<type_real>(1000.0);
  water.p_velocity = static_cast<type_real>(1500.0);
  water.thickness = static_cast<type_real>(30000.0);
  fk_injection::ElasticIsotropicLayer solid;
  solid.density = static_cast<type_real>(2700.0);
  solid.p_velocity = static_cast<type_real>(3000.0);
  solid.s_velocity = static_cast<type_real>(1732.051);
  const double p = std::sin(20.0 * M_PI / 180.0) / 3000.0;

  const double expected = 39.41077 * std::log(1.0e-4) / std::log(0.6816769);
  EXPECT_NEAR(fk_injection::impl::fluid_reverberation_time(
                  fk_injection::LayeredModel({ water, solid }), p),
              expected, 1.0e-3 * expected);
  EXPECT_EQ(fk_injection::impl::fluid_reverberation_time(
                fk_injection::LayeredModel({ solid }), p),
            0.0);
}
