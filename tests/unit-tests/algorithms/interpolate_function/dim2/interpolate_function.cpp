
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "specfem/algorithms/interpolate.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/shape_function.hpp"
#include <Kokkos_Core.hpp>
/*
 * Square element with 4 nodes
 * 4---3
 * |   |
 * 1---2
 */
static std::array<std::array<type_real, 2>, 4> square_4node_coords = {
  { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 } }
};

/*
 * Square element with 9 nodes
 * 4---7---3
 * |   |   |
 * 8---9---6
 * |   |   |
 * 1---5---2
 */
static std::array<std::array<type_real, 2>, 9> square_9node_coords = {
  { { 0.0, 0.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 },
    { 0.0, 1.0 },
    { 0.5, 0.0 },
    { 1.0, 0.5 },
    { 0.5, 1.0 },
    { 0.0, 0.5 },
    { 0.5, 0.5 } }
};
/* * Distorted 4-node element
 *   4---3
 *  /   /
 * 1---2
 */
static std::array<std::array<type_real, 2>, 4> distorted_4node_coords = {
  { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.2, 1.0 }, { 0.2, 1.0 } }
};
/* * Distorted 9-node element
 *       4---7---3
 *      /   /   /
 *     8---9---6
 *    /   /   /
 *   1---5---2
 */
static std::array<std::array<type_real, 2>, 9> distorted_9node_coords = {
  { { 0.0, 0.0 },
    { 1.0, 0.0 },
    { 1.2, 1.0 },
    { 0.2, 1.0 },
    { 0.5, 0.0 },
    { 1.1, 0.5 },
    { 0.7, 1.0 },
    { 0.1, 0.5 },
    { 0.6, 0.5 } }
};

Kokkos::View<type_real **, Kokkos::HostSpace> compute_interpolant_polynomial(
    const type_real xi_target, const type_real gamma_target, const int N,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &xi) {
  auto [hxi, hpxi] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          xi_target, N, xi);
  auto [hgamma, hpgamma] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          gamma_target, N, xi);

  Kokkos::View<type_real **, Kokkos::HostSpace> polynomial("polynomial", N, N);
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      polynomial(iz, ix) = hxi(ix) * hxi(iz);
    }
  }

  return polynomial;
}

class SingleElementMesh {
public:
  SingleElementMesh() {}

  int ngnod; ///< Number of control nodes
  SingleElementMesh(int ngnod)
      : ngnod(ngnod), Coordinates("Coordinates", ngnod) {}

  template <typename Generator>
  SingleElementMesh(int ngnod, const Generator &gen)
      : ngnod(ngnod), Coordinates("Coordinates", ngnod) {
    for (int in = 0; in < ngnod; ++in) {
      Coordinates(in, 0) = gen(in, 0);
      Coordinates(in, 1) = gen(in, 1);
    }
  }

  Kokkos::View<type_real *[2], Kokkos::HostSpace> Coordinates;

  std::tuple<type_real, type_real> get_coordinates(type_real xi,
                                                   type_real gamma) const {
    const auto shape2D =
        specfem::shape_function::shape_function(xi, gamma, ngnod);
    type_real xcor = 0.0;
    type_real zcor = 0.0;

    for (int in = 0; in < ngnod; in++) {
      xcor += shape2D[in] * Coordinates(in, 0);
      zcor += shape2D[in] * Coordinates(in, 1);
    }

    return { xcor, zcor };
  }
};

class InterpolateFunction2D : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary data or state before each test
  }

  void TearDown() override {
    // Clean up any resources after each test
  }

  SingleElementMesh square_4node_element;

  SingleElementMesh square_9node_element;

  SingleElementMesh distorted_4node_element;

  SingleElementMesh distorted_9node_element;

  InterpolateFunction2D()
      : square_4node_element(
            4,
            [](int ngnod, int dim) { return square_4node_coords[ngnod][dim]; }),
        square_9node_element(
            9,
            [](int ngnod, int dim) { return square_9node_coords[ngnod][dim]; }),
        distorted_4node_element(4,
                                [](int ngnod, int dim) {
                                  return distorted_4node_coords[ngnod][dim];
                                }),
        distorted_9node_element(9, [](int ngnod, int dim) {
          return distorted_9node_coords[ngnod][dim];
        }) {}
};

TEST_F(InterpolateFunction2D, LinearFunction4NodeElement) {
  const int NGLL = 5;
  // Simple linear function
  const auto function = [](type_real x, type_real z) { return x + z; };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at the center of the element
  const type_real xi_target = 0.5;
  const type_real gamma_target = 0.5;

  {
    auto [x, z] = square_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();

    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in 4-node Square element";
  }

  {
    auto [x, z] =
        distorted_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in distorted 4-node "
           "element";
  }
}

TEST_F(InterpolateFunction2D, LinearFunction9NodeElement) {
  const int NGLL = 5;
  // Simple linear function
  const auto function = [](type_real x, type_real z) { return x + z; };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at the center of the element
  const type_real xi_target = 0.25;
  const type_real gamma_target = 0.25;

  {
    auto [x, z] = square_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in 9-node Square element";
  }

  {
    auto [x, z] =
        distorted_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);
    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in distorted 9-node "
           "element";
  }
}

TEST_F(InterpolateFunction2D, QuadraticFunction4NodeElement) {
  const int NGLL = 5;
  // Simple quadratic function
  const auto function = [](type_real x, type_real z) { return x * x + z * z; };
  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at the center of the element
  const type_real xi_target = 0.5;
  const type_real gamma_target = 0.5;

  {
    auto [x, z] = square_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in 4-node Square "
           "element";
  }

  {
    auto [x, z] =
        distorted_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in distorted 4-node "
           "element";
  }
}

TEST_F(InterpolateFunction2D, QuadraticFunction9NodeElement) {
  const int NGLL = 5;
  // Simple quadratic function
  const auto function = [](type_real x, type_real z) { return x * x + z * z; };
  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at the center of the element
  const type_real xi_target = 0.25;
  const type_real gamma_target = 0.25;

  {
    auto [x, z] = square_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in 9-node Square "
           "element";
  }

  {
    auto [x, z] =
        distorted_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in distorted 9-node "
           "element";
  }
}

TEST_F(InterpolateFunction2D, ConstantFunction) {
  const int NGLL = 3; // Lower order sufficient for constant
  // Constant function
  const auto function = [](type_real x, type_real z) { return 42.0; };

  // Generate 3 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at any point
  const type_real xi_target = 0.8;
  const type_real gamma_target = 0.2;

  {
    auto [x, z] = square_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-3)
        << "Interpolation failed for constant function in 4-node Square "
           "element";
  }

  {
    auto [x, z] = square_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = square_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-3)
        << "Interpolation failed for constant function in 9-node Square "
           "element";
  }

  {
    auto [x, z] =
        distorted_4node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_4node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-3)
        << "Interpolation failed for constant function in distorted 4-node "
           "element";
  }

  {
    auto [x, z] =
        distorted_9node_element.get_coordinates(xi_target, gamma_target);
    type_real function_value = function(x, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial =
        compute_interpolant_polynomial(xi_target, gamma_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real **, Kokkos::HostSpace> function_view("function",
                                                                NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int ix = 0; ix < NGLL; ++ix) {
        auto [x, z] = distorted_9node_element.get_coordinates(hxi(ix), hxi(iz));
        function_view(iz, ix) = function(x, z);
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-3)
        << "Interpolation failed for constant function in distorted 9-node "
           "element";
  }
}
