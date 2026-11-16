#include "fk_injection/incident_wave.hpp"
#include <cmath>
#include <gtest/gtest.h>

using fk_injection::incident_wave_type;
using fk_injection::IncidentWave;

namespace incident_wave_tests_impl {

// A half-space with vp = 8000 m/s and vs = 4500 m/s.
inline fk_injection::LayeredModel make_halfspace_model() {
  fk_injection::ElasticIsotropicLayer halfspace;
  halfspace.density = 3300.0;
  halfspace.p_velocity = 8000.0;
  halfspace.s_velocity = 4500.0;
  return fk_injection::LayeredModel({ halfspace });
}

} // namespace incident_wave_tests_impl

// theta is stored in type_real, which may be float.
static constexpr double kTol = 1.0e-9;

// ---------------------------------------------------------------------------
// ray_parameter = sin(theta) / halfspace_velocity of the wave's type
// ---------------------------------------------------------------------------

TEST(IncidentWave, RayParameter) {
  IncidentWave wave;
  wave.type = incident_wave_type::p;
  wave.take_off_theta = static_cast<fk_injection::type_real>(M_PI / 6.0); // 30°

  // sin(pi/6) = 0.5, so ray_parameter = 0.5 / 8000.
  EXPECT_NEAR(
      wave.ray_parameter(incident_wave_tests_impl::make_halfspace_model()),
      0.5 / 8000.0, kTol);
}

TEST(IncidentWave, RayParameterSvUsesShearVelocity) {
  IncidentWave wave;
  wave.type = incident_wave_type::sv;
  wave.take_off_theta = static_cast<fk_injection::type_real>(M_PI / 6.0); // 30°

  // sin(pi/6) = 0.5, so ray_parameter = 0.5 / 4500.
  EXPECT_NEAR(
      wave.ray_parameter(incident_wave_tests_impl::make_halfspace_model()),
      0.5 / 4500.0, kTol);
}

TEST(IncidentWave, RayParameterZeroAngle) {
  IncidentWave wave;
  wave.take_off_theta = 0.0;

  // Vertical incidence: sin(0) = 0 → ray_parameter == 0.
  EXPECT_NEAR(
      wave.ray_parameter(incident_wave_tests_impl::make_halfspace_model()), 0.0,
      kTol);
}

TEST(IncidentWave, RayParameterNinetyDegrees) {
  IncidentWave wave;
  wave.take_off_theta = static_cast<fk_injection::type_real>(M_PI / 2.0); // 90°

  // sin(pi/2) = 1 → ray_parameter = 1/8000.
  EXPECT_NEAR(
      wave.ray_parameter(incident_wave_tests_impl::make_halfspace_model()),
      1.0 / 8000.0, kTol);
}
