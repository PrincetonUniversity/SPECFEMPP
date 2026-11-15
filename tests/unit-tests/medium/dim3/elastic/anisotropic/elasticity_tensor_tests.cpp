#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"

#include "globe_model_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace specfem::medium_physics::elasticity_tensor_test_impl {

using Cartesian = specfem::coordinate_systems::CartesianFrame;
using Geocentric = specfem::coordinate_systems::GeocentricFrame;
using Tensor = specfem::medium_physics::ElasticityTensor<Cartesian, double>;
using Radial = specfem::medium_physics::ElasticityTensor<Geocentric, double>;
using Forward =
    specfem::coordinate_systems::BasisMatrix<Cartesian, Geocentric, double>;
using Inverse =
    specfem::coordinate_systems::BasisMatrix<Geocentric, Cartesian, double>;

template <typename TensorType, typename MatrixType>
concept CanRotate =
    requires(const TensorType &tensor, const MatrixType &matrix) {
      specfem::medium_physics::rotate(tensor, matrix);
    };

static_assert(CanRotate<Radial, Forward>);
static_assert(CanRotate<Tensor, Inverse>);
static_assert(!CanRotate<Tensor, Forward>);
static_assert(!CanRotate<Radial, Inverse>);
static_assert(
    std::is_same_v<decltype(specfem::medium_physics::rotate(
                       std::declval<Radial>(), std::declval<Forward>())),
                   Tensor>);
static_assert(std::is_constructible_v<Radial, double, double, double, double,
                                      double, double>);
static_assert(!std::is_constructible_v<Tensor, double, double, double, double,
                                       double, double>);
static_assert(!std::is_convertible_v<Radial, Tensor>);
static_assert(!std::is_assignable_v<Tensor &, Radial>);
static_assert(std::is_default_constructible_v<Radial>);
static_assert(std::is_trivially_copyable_v<Radial>);
static_assert(std::is_trivially_copyable_v<Tensor>);
static_assert(std::is_trivially_copyable_v<
              specfem::medium_physics::ElasticityTensor<Cartesian, float>>);
static_assert(std::is_trivially_copyable_v<
              specfem::medium_physics::ElasticityTensor<Geocentric, float>>);
static_assert(sizeof(Tensor) == 21 * sizeof(double));

constexpr std::array<std::array<int, 2>, 21> component_indices = {
  { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 1 },
    { 1, 2 }, { 1, 3 }, { 1, 4 }, { 1, 5 }, { 2, 2 }, { 2, 3 }, { 2, 4 },
    { 2, 5 }, { 3, 3 }, { 3, 4 }, { 3, 5 }, { 4, 4 }, { 4, 5 }, { 5, 5 } }
};

template <typename Frame>
void expect_tensor_near(
    const specfem::medium_physics::ElasticityTensor<Frame, double> &actual,
    const specfem::medium_physics::ElasticityTensor<Frame, double> &expected,
    const double relative_tolerance = 2.e-12) {
  double tensor_scale = 1.0;
  for (int component = 0; component < 21; ++component) {
    tensor_scale = std::max({ tensor_scale, std::abs(actual[component]),
                              std::abs(expected[component]) });
  }
  for (int component = 0; component < 21; ++component) {
    EXPECT_NEAR(actual[component], expected[component],
                relative_tolerance * tensor_scale)
        << "component " << component;
  }
}

struct DeviceResults {
  Tensor global;
  Radial round_trip;
};

DeviceResults rotate_on_device() {
  Kokkos::View<Tensor *> global("global_elasticity", 1);
  Kokkos::View<Radial *> round_trip("radial_elasticity", 1);
  Kokkos::parallel_for(
      "rotate_on_device", 1, KOKKOS_LAMBDA(const int) {
        const Radial radial(4.0, 3.0, 5.0, 2.0, 2.5, 0.8);
        double packed[21];
        for (int k = 0; k < 21; ++k) {
          packed[k] = radial[k];
        }
        const Radial from_packed(packed);
        const auto rotation =
            specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(
                0.7, 1.1);
        global(0) = specfem::medium_physics::rotate(from_packed, rotation);
        const auto inverse =
            specfem::coordinate_systems::RotationMatrix<Geocentric,
                                                        Cartesian>{}(0.7, 1.1);
        round_trip(0) = specfem::medium_physics::rotate(global(0), inverse);
      });
  const auto host_global =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, global);
  const auto host_radial =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, round_trip);
  return { host_global(0), host_radial(0) };
}

TEST(ElasticityTensor, PackedConstructionAndSymmetricAccess) {
  double coefficients[21];
  for (int k = 0; k < 21; ++k) {
    coefficients[k] = k + 0.5;
  }
  Tensor tensor(coefficients);
  const Tensor &constant = tensor;
  for (int k = 0; k < 21; ++k) {
    const auto [row, column] = component_indices[k];
    EXPECT_DOUBLE_EQ(constant[k], coefficients[k]);
    EXPECT_DOUBLE_EQ(constant(row, column), coefficients[k]);
    EXPECT_DOUBLE_EQ(constant(column, row), coefficients[k]);
    tensor(column, row) = -coefficients[k];
    EXPECT_DOUBLE_EQ(tensor(row, column), -coefficients[k]);
  }
  const Radial zero;
  for (int k = 0; k < 21; ++k) {
    EXPECT_DOUBLE_EQ(zero[k], 0.0);
  }
}

TEST(ElasticityTensor, LoveParametersConstructRadialTensor) {
  constexpr double rho = 4.0;
  constexpr double vpv = 3.0;
  constexpr double vph = 5.0;
  constexpr double vsv = 2.0;
  constexpr double vsh = 2.5;
  constexpr double eta = 0.8;

  const Radial tensor(rho, vpv, vph, vsv, vsh, eta);
  const double a = rho * vph * vph;
  const double c = rho * vpv * vpv;
  const double n = rho * vsh * vsh;
  const double l = rho * vsv * vsv;
  const double f = eta * (a - 2.0 * l);
  const double expected_coefficients[] = { a,   a - 2.0 * n, f,   0.0, 0.0, 0.0,
                                           a,   f,           0.0, 0.0, 0.0, c,
                                           0.0, 0.0,         0.0, l,   0.0, 0.0,
                                           l,   0.0,         n };
  const Radial expected(expected_coefficients);
  expect_tensor_near(tensor, expected, 0.0);
}

TEST(ElasticityTensor, LoveRotationMatchesGlobeReference) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const std::array<std::array<double, 2>, 8> angles = { {
      { 0.0, 0.0 },
      { 0.0, 1.7 },
      { pi, 0.0 },
      { pi, 5.2 },
      { pi / 2.0, 0.0 },
      { pi / 2.0, pi / 2.0 },
      { 0.37, 2.91 },
      { 2.63, 6.1 },
  } };

  for (int sample = 0; sample < 8; ++sample) {
    const double rho = 2.5 + 0.2 * sample;
    const double vpv = 3.0 + 0.11 * sample;
    const double vph = 4.2 + 0.17 * sample;
    const double vsv = 1.7 + 0.07 * sample;
    const double vsh = 2.1 + 0.09 * sample;
    const double eta = 0.7 + 0.04 * sample;
    const Radial radial(rho, vpv, vph, vsv, vsh, eta);

    for (const auto &[theta, phi] : angles) {
      double globe[21] = {};
      globe_evaluator_rotate_tiso_to_cij(theta, phi, rho, vpv, vph, vsv, vsh,
                                         eta, globe);
      const auto actual = specfem::medium_physics::rotate(
          radial,
          specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(
              theta, phi));
      expect_tensor_near(actual, Tensor(globe), 1.e-10);
    }
  }
}

TEST(ElasticityTensor, GlobalRadialRoundTrip) {
  const double input_coefficients[] = { 31.0, 2.0,  3.0,  4.0,  5.0,  6.0,
                                        37.0, 8.0,  9.0,  10.0, 11.0, 41.0,
                                        13.0, 14.0, 15.0, 43.0, 17.0, 18.0,
                                        47.0, 20.0, 53.0 };
  const Tensor input(input_coefficients);
  constexpr double pi = 3.141592653589793238462643383279502884;
  const std::array<std::array<double, 2>, 7> angles = { {
      { 0.0, 0.0 },
      { 0.0, 1.7 },
      { pi, 0.0 },
      { pi, 5.2 },
      { pi / 2.0, 0.0 },
      { 0.37, 2.91 },
      { 2.63, 6.1 },
  } };

  for (const auto &[theta, phi] : angles) {
    const auto radial = specfem::medium_physics::rotate(
        input, specfem::coordinate_systems::rotation_matrix<
                   specfem::coordinate_systems::GeocentricFrame,
                   specfem::coordinate_systems::CartesianFrame>(theta, phi));
    const auto round_trip = specfem::medium_physics::rotate(
        radial,
        specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(
            theta, phi));
    expect_tensor_near(round_trip, input, 8.e-12);
  }
}

TEST(ElasticityTensor, ArbitraryRotationMatchesFourthOrderReference) {
  const double input_coefficients[] = { 31.0, 2.0,  3.0,  4.0,  5.0,  6.0,
                                        37.0, 8.0,  9.0,  10.0, 11.0, 41.0,
                                        13.0, 14.0, 15.0, 43.0, 17.0, 18.0,
                                        47.0, 20.0, 53.0 };
  const Tensor input(input_coefficients);
  // Rodrigues rotation about (2, -1, 2)/3 includes an independent third angle.
  const double axis[3] = { 2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0 };
  const double skew[3][3] = { { 0.0, -axis[2], axis[1] },
                              { axis[2], 0.0, -axis[0] },
                              { -axis[1], axis[0], 0.0 } };
  Inverse rotation{};
  const double cosine = std::cos(0.63);
  const double sine = std::sin(0.63);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation(row, column) = (row == column ? cosine : 0.0) +
                              (1.0 - cosine) * axis[row] * axis[column] +
                              sine * skew[row][column];
    }
  }

  double stiffness[6][6] = {};
  for (int component = 0; component < 21; ++component) {
    const auto [row, column] = component_indices[component];
    stiffness[row][column] = input[component];
    stiffness[column][row] = input[component];
  }
  // Expand using tensor symmetries and contract all four indices directly.
  constexpr int voigt[3][3] = { { 0, 5, 4 }, { 5, 1, 3 }, { 4, 3, 2 } };
  double expected[3][3][3][3] = {};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              for (int c = 0; c < 3; ++c) {
                for (int d = 0; d < 3; ++d) {
                  expected[i][j][k][l] += rotation(i, a) * rotation(j, b) *
                                          rotation(k, c) * rotation(l, d) *
                                          stiffness[voigt[a][b]][voigt[c][d]];
                }
              }
            }
          }
        }
      }
    }
  }
  constexpr int pairs[6][2] = { { 0, 0 }, { 1, 1 }, { 2, 2 },
                                { 1, 2 }, { 0, 2 }, { 0, 1 } };
  Radial reference{};
  for (int component = 0; component < 21; ++component) {
    const auto [row, column] = component_indices[component];
    reference(row, column) = expected[pairs[row][0]][pairs[row][1]]
                                     [pairs[column][0]][pairs[column][1]];
  }
  expect_tensor_near(specfem::medium_physics::rotate(input, rotation),
                     reference);
}

TEST(ElasticityTensor, RotationIsDeviceCallable) {
  const auto result = rotate_on_device();
  EXPECT_NEAR(result.global[3], 0.55226814535471225, 2.e-13);
  EXPECT_NEAR(result.global[13], -7.8509057040399073, 2.e-13);
  const Radial radial(4.0, 3.0, 5.0, 2.0, 2.5, 0.8);
  expect_tensor_near(result.round_trip, radial);
}

TEST(ElasticityTensor, IsotropicLoveTensorIsRotationInvariant) {
  constexpr double rho = 3300.0;
  constexpr double vp = 8000.0;
  constexpr double vs = 4500.0;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double mu = rho * vs * vs;
  const double c11 = rho * vp * vp;
  const double c12 = c11 - 2.0 * mu;
  const double isotropic_coefficients[] = { c11, c12, c12, 0.0, 0.0, 0.0, c11,
                                            c12, 0.0, 0.0, 0.0, c11, 0.0, 0.0,
                                            0.0, mu,  0.0, 0.0, mu,  0.0, mu };
  const Tensor isotropic(isotropic_coefficients);
  const Radial radial(rho, vp, vp, vs, vs, 1.0);

  for (const auto &[theta, phi] : std::array<std::array<double, 2>, 4>{
           { { 0.0, 0.0 }, { pi, 4.3 }, { pi / 2.0, 0.7 }, { 1.1, 5.4 } } }) {
    const auto global = specfem::medium_physics::rotate(
        radial,
        specfem::coordinate_systems::rotation_matrix<Cartesian, Geocentric>(
            theta, phi));
    expect_tensor_near(global, isotropic, 3.e-15);

    double expanded[6][6] = {};
    for (int component = 0; component < 21; ++component) {
      const auto [row, column] = component_indices[component];
      expanded[row][column] = global[component];
      expanded[column][row] = global[component];
    }
    for (int row = 0; row < 6; ++row) {
      for (int column = 0; column < 6; ++column) {
        EXPECT_DOUBLE_EQ(expanded[row][column], expanded[column][row]);
      }
    }
  }
}

} // namespace specfem::medium_physics::elasticity_tensor_test_impl
