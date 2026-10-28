#include "specfem/coordinate_systems/east_north_up.hpp"

#include <gtest/gtest.h>

namespace specfem::coordinate_systems::east_north_up_test_impl {

// East = +phi, North = -theta, Up = +r: (E, N, U) -> (theta, phi, r) is
// (-N, E, U). This is the one place the North sign flip is asserted.
TEST(EastNorthUp, ReordersIntoGeocentricComponents) {
  const specfem::coordinate_systems::EastNorthUp<double> enu{ 1.0, 2.0, 3.0 };
  const auto geocentric =
      specfem::coordinate_systems::to_geocentric_components(enu);
  EXPECT_DOUBLE_EQ(geocentric[0], -2.0); // theta = -North
  EXPECT_DOUBLE_EQ(geocentric[1], 1.0);  // phi = East
  EXPECT_DOUBLE_EQ(geocentric[2], 3.0);  // r = Up
}

TEST(EastNorthUp, RoundTrips) {
  const specfem::coordinate_systems::EastNorthUp<float> enu{ -4.0f, 0.5f,
                                                             7.0f };
  const auto back = specfem::coordinate_systems::to_east_north_up_components(
      specfem::coordinate_systems::to_geocentric_components(enu));
  EXPECT_FLOAT_EQ(back.east, enu.east);
  EXPECT_FLOAT_EQ(back.north, enu.north);
  EXPECT_FLOAT_EQ(back.up, enu.up);

  const Kokkos::Array<double, 3> theta_phi_r{ 0.25, -1.5, 9.0 };
  const auto twice = specfem::coordinate_systems::to_geocentric_components(
      specfem::coordinate_systems::to_east_north_up_components(theta_phi_r));
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(twice[i], theta_phi_r[i]);
  }
}

} // namespace specfem::coordinate_systems::east_north_up_test_impl
