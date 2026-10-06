#include "specfem/injection/fk/incident_wave.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::injection::fk::incident_wave_type;
using specfem::injection::fk::IncidentWave;

// type_real may be float; use a tolerance appropriate for single precision.
static constexpr double kTol = 1.0e-6;

// ---------------------------------------------------------------------------
// ray_parameter = sin(theta) / halfspace_velocity
// ---------------------------------------------------------------------------

TEST(IncidentWave, RayParameter) {
  IncidentWave wave;
  wave.type = incident_wave_type::p;
  wave.take_off_theta = static_cast<type_real>(M_PI / 6.0); // 30°

  const type_real halfspace_velocity = 8000.0;

  // sin(pi/6) = 0.5, so ray_parameter = 0.5 / 8000.
  const double expected = 0.5 / 8000.0;

  EXPECT_NEAR(static_cast<double>(wave.ray_parameter(halfspace_velocity)),
              expected, kTol);
}

TEST(IncidentWave, RayParameterZeroAngle) {
  IncidentWave wave;
  wave.take_off_theta = 0.0;

  // Vertical incidence: sin(0) = 0 → ray_parameter == 0.
  EXPECT_NEAR(static_cast<double>(wave.ray_parameter(6000.0)), 0.0, kTol);
}

TEST(IncidentWave, RayParameterNinetyDegrees) {
  IncidentWave wave;
  wave.take_off_theta = static_cast<type_real>(M_PI / 2.0); // 90°

  const type_real halfspace_velocity = 5000.0;
  // sin(pi/2) = 1 → ray_parameter = 1/5000.
  EXPECT_NEAR(static_cast<double>(wave.ray_parameter(halfspace_velocity)),
              1.0 / 5000.0, kTol);
}
