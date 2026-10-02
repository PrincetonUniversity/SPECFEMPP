#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace specfem::medium_physics::elasticity_tensor_test_impl {

using Tensor = specfem::medium_physics::elasticity_tensor<double>;

constexpr std::array<std::array<int, 2>, 21> component_indices = {
  { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 }, { 0, 5 }, { 1, 1 },
    { 1, 2 }, { 1, 3 }, { 1, 4 }, { 1, 5 }, { 2, 2 }, { 2, 3 }, { 2, 4 },
    { 2, 5 }, { 3, 3 }, { 3, 4 }, { 3, 5 }, { 4, 4 }, { 4, 5 }, { 5, 5 } }
};

void expect_tensor_near(const Tensor &actual, const Tensor &expected,
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

TEST(ElasticityTensor, LoveParametersConstructRadialTensor) {
  constexpr double rho = 4.0;
  constexpr double vpv = 3.0;
  constexpr double vph = 5.0;
  constexpr double vsv = 2.0;
  constexpr double vsh = 2.5;
  constexpr double eta = 0.8;

  const auto tensor = specfem::medium_physics::love_to_radial_elasticity(
      rho, vpv, vph, vsv, vsh, eta);
  const double a = rho * vph * vph;
  const double c = rho * vpv * vpv;
  const double n = rho * vsh * vsh;
  const double l = rho * vsv * vsv;
  const double f = eta * (a - 2.0 * l);
  const Tensor expected = { a,   a - 2.0 * n, f,   0.0, 0.0, 0.0, a,
                            f,   0.0,         0.0, 0.0, c,   0.0, 0.0,
                            0.0, l,           0.0, 0.0, l,   0.0, n };
  expect_tensor_near(tensor, expected, 0.0);
}

TEST(ElasticityTensor, LoveRotationMatchesIndependentReference) {
  const auto radial = specfem::medium_physics::love_to_radial_elasticity(
      4.0, 3.0, 5.0, 2.0, 2.5, 0.8);
  const auto global =
      specfem::medium_physics::rotate_elasticity_radial_to_global(radial, 0.7,
                                                                  1.1);
  // Independently evaluated as C_ijkl = R_ip R_jq R_kr R_ls D_pqrs.
  const Tensor expected = {
    97.409086709298677,  50.790275426336144,  51.111429723630465,
    0.55226814535471225, -3.7418787069187522, -2.8088547369406829,
    87.035678349151596,  46.92827522367427,   -11.298678904814622,
    -1.7277016154442508, -4.3167590542834375, 71.49527419426812,
    -15.425142800160938, -7.8509057040399073, -2.8734570510329767,
    9.6724931613562362,  -5.1215702940907812, -4.7225759905079929,
    17.128431928516854,  -5.3319402052983005, 20.229055283767746
  };
  expect_tensor_near(global, expected, 2.e-15);
}

TEST(ElasticityTensor, GlobalRadialRoundTrip) {
  const Tensor input = { 31.0, 2.0,  3.0,  4.0,  5.0,  6.0,  37.0,
                         8.0,  9.0,  10.0, 11.0, 41.0, 13.0, 14.0,
                         15.0, 43.0, 17.0, 18.0, 47.0, 20.0, 53.0 };
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
    const auto radial =
        specfem::medium_physics::rotate_elasticity_global_to_radial(input,
                                                                    theta, phi);
    const auto round_trip =
        specfem::medium_physics::rotate_elasticity_radial_to_global(radial,
                                                                    theta, phi);
    expect_tensor_near(round_trip, input, 8.e-12);
  }
}

TEST(ElasticityTensor, RotationIsDeviceCallable) {
  const bool initialize_here = !Kokkos::is_initialized();
  if (initialize_here) {
    Kokkos::initialize();
  }
  {
    Kokkos::View<Tensor *> result("rotated_elasticity", 1);
    Kokkos::parallel_for(
        "rotate_elasticity_on_device", 1, KOKKOS_LAMBDA(const int) {
          const auto radial =
              specfem::medium_physics::love_to_radial_elasticity(4.0, 3.0, 5.0,
                                                                 2.0, 2.5, 0.8);
          result(0) =
              specfem::medium_physics::rotate_elasticity_radial_to_global(
                  radial, 0.7, 1.1);
        });
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
    EXPECT_NEAR(host(0)[3], 0.55226814535471225, 2.e-13);
    EXPECT_NEAR(host(0)[13], -7.8509057040399073, 2.e-13);
  }
  if (initialize_here) {
    Kokkos::finalize();
  }
}

TEST(ElasticityTensor, IsotropicLoveTensorIsRotationInvariant) {
  constexpr double rho = 3300.0;
  constexpr double vp = 8000.0;
  constexpr double vs = 4500.0;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double mu = rho * vs * vs;
  const double c11 = rho * vp * vp;
  const double c12 = c11 - 2.0 * mu;
  const Tensor isotropic = { c11, c12, c12, 0.0, 0.0, 0.0, c11,
                             c12, 0.0, 0.0, 0.0, c11, 0.0, 0.0,
                             0.0, mu,  0.0, 0.0, mu,  0.0, mu };
  const auto radial = specfem::medium_physics::love_to_radial_elasticity(
      rho, vp, vp, vs, vs, 1.0);

  for (const auto &[theta, phi] : std::array<std::array<double, 2>, 4>{
           { { 0.0, 0.0 }, { pi, 4.3 }, { pi / 2.0, 0.7 }, { 1.1, 5.4 } } }) {
    const auto global =
        specfem::medium_physics::rotate_elasticity_radial_to_global(radial,
                                                                    theta, phi);
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
