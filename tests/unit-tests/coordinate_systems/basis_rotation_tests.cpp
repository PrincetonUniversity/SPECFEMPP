#include "specfem/coordinate_systems/basis_rotation.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace specfem::coordinate_systems::basis_rotation_test_impl {

TEST(BasisRotation, MapsLocalUnitAxesAtKnownLocations) {
  constexpr double half_pi = 1.57079632679489661923;
  struct Sample {
    double theta;
    double phi;
    specfem::coordinate_systems::rotation_matrix<double> expected;
  };
  // Columns give the known directions of (theta, phi, r), respectively.
  const std::array<Sample, 4> samples = { {
      { 0.0,
        0.0,
        { { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } } },
      { half_pi,
        0.0,
        { { { 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0 }, { -1.0, 0.0, 0.0 } } } },
      { half_pi,
        half_pi,
        { { { 0.0, -1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { -1.0, 0.0, 0.0 } } } },
      { 0.0,
        half_pi,
        { { { 0.0, -1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 } } } },
  } };
  for (const auto &sample : samples) {
    const auto rotation = specfem::coordinate_systems::basis_rotation<
        specfem::coordinate_systems::CartesianFrame,
        specfem::coordinate_systems::GeocentricFrame>(sample.theta, sample.phi);
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        EXPECT_NEAR(rotation[row][column], sample.expected[row][column],
                    2.e-15);
      }
    }
  }
}

TEST(BasisRotation, InverseIsTransposeAndOrthonormalAtPoles) {
  constexpr double pi = 3.14159265358979323846;
  for (const double theta : { 0.0, 0.7, pi / 2.0, pi }) {
    for (const double phi : { 0.0, 1.1, 5.2 }) {
      const auto forward = specfem::coordinate_systems::basis_rotation<
          specfem::coordinate_systems::CartesianFrame,
          specfem::coordinate_systems::GeocentricFrame>(theta, phi);
      const auto inverse = specfem::coordinate_systems::basis_rotation<
          specfem::coordinate_systems::GeocentricFrame,
          specfem::coordinate_systems::CartesianFrame>(theta, phi);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          EXPECT_DOUBLE_EQ(inverse[row][column], forward[column][row]);
          double product = 0.0;
          for (int inner = 0; inner < 3; ++inner) {
            product += inverse[row][inner] * forward[inner][column];
          }
          EXPECT_NEAR(product, row == column ? 1.0 : 0.0, 2.e-15);
        }
      }
      // A radial unit vector always maps to the geometric radial direction.
      EXPECT_NEAR(forward[2][2], std::cos(theta), 2.e-15);
    }
  }
}

TEST(BasisRotation, SupportsFloatScalars) {
  const auto rotation = specfem::coordinate_systems::basis_rotation<
      specfem::coordinate_systems::CartesianFrame,
      specfem::coordinate_systems::GeocentricFrame>(0.0f, 0.0f);
  const specfem::coordinate_systems::rotation_matrix<float> identity = {
    { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
  };
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      EXPECT_FLOAT_EQ(rotation[row][column], identity[row][column]);
    }
  }
}

} // namespace specfem::coordinate_systems::basis_rotation_test_impl
