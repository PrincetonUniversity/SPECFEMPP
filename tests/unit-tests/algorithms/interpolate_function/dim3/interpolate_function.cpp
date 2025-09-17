#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "algorithms/interpolate.hpp"
#include "quadrature/interface.hpp"
#include "specfem/shape_functions.hpp"
#include <Kokkos_Core.hpp>

/*
 * Cube element with 8 nodes
 *     8-------7
 *    /|      /|
 *   5-------6 |
 *   | |     | |
 *   | 4-----|-3
 *   |/      |/
 *   1-------2
 */
static std::array<std::array<type_real, 3>, 8> cube_8node_coords = { {
    { 0.0, 0.0, 0.0 }, // 1
    { 1.0, 0.0, 0.0 }, // 2
    { 1.0, 1.0, 0.0 }, // 3
    { 0.0, 1.0, 0.0 }, // 4
    { 0.0, 0.0, 1.0 }, // 5
    { 1.0, 0.0, 1.0 }, // 6
    { 1.0, 1.0, 1.0 }, // 7
    { 0.0, 1.0, 1.0 }  // 8
} };

/*
 * Cube element with 27 nodes (3x3x3 grid)
 * Similar to 8-node but with additional nodes on edges, faces, and center
 */
static std::array<std::array<type_real, 3>, 27> cube_27node_coords = {
  { // Corner nodes (1-8)
    { 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 1.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 },
    { 1.0, 0.0, 1.0 },
    { 1.0, 1.0, 1.0 },
    { 0.0, 1.0, 1.0 },
    // Edge nodes (9-20)
    { 0.5, 0.0, 0.0 },
    { 1.0, 0.5, 0.0 },
    { 0.5, 1.0, 0.0 },
    { 0.0, 0.5, 0.0 },
    { 0.5, 0.0, 1.0 },
    { 1.0, 0.5, 1.0 },
    { 0.5, 1.0, 1.0 },
    { 0.0, 0.5, 1.0 },
    { 0.0, 0.0, 0.5 },
    { 1.0, 0.0, 0.5 },
    { 1.0, 1.0, 0.5 },
    { 0.0, 1.0, 0.5 },
    // Face centers (21-26)
    { 0.5, 0.5, 0.0 },
    { 0.5, 0.5, 1.0 },
    { 0.5, 0.0, 0.5 },
    { 1.0, 0.5, 0.5 },
    { 0.5, 1.0, 0.5 },
    { 0.0, 0.5, 0.5 },
    // Volume center (27)
    { 0.5, 0.5, 0.5 } }
};

/*
 * Distorted 8-node element
 *     8-------7
 *    /|      /|
 *   5-------6 |
 *   | |     | |
 *   | 4-----|-3
 *   |/      |/
 *   1-------2
 */
static std::array<std::array<type_real, 3>, 8> distorted_8node_coords = {
  { { 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.2, 1.0, 0.0 },
    { 0.2, 1.0, 0.0 },
    { 0.1, 0.1, 1.0 },
    { 1.1, 0.1, 1.0 },
    { 1.3, 1.1, 1.0 },
    { 0.3, 1.1, 1.0 } }
};

/*
 * Distorted 27-node element
 */
static std::array<std::array<type_real, 3>, 27> distorted_27node_coords = {
  { // Corner nodes (1-8)
    { 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.2, 1.0, 0.0 },
    { 0.2, 1.0, 0.0 },
    { 0.1, 0.1, 1.0 },
    { 1.1, 0.1, 1.0 },
    { 1.3, 1.1, 1.0 },
    { 0.3, 1.1, 1.0 },
    // Edge nodes (approximate midpoints)
    { 0.5, 0.0, 0.0 },
    { 1.1, 0.5, 0.0 },
    { 0.7, 1.0, 0.0 },
    { 0.1, 0.5, 0.0 },
    { 0.6, 0.1, 1.0 },
    { 1.2, 0.6, 1.0 },
    { 0.8, 1.1, 1.0 },
    { 0.2, 0.6, 1.0 },
    { 0.05, 0.05, 0.5 },
    { 1.05, 0.05, 0.5 },
    { 1.25, 1.05, 0.5 },
    { 0.25, 1.05, 0.5 },
    // Face centers
    { 0.6, 0.5, 0.0 },
    { 0.7, 0.6, 1.0 },
    { 0.55, 0.05, 0.5 },
    { 1.15, 0.55, 0.5 },
    { 0.75, 1.05, 0.5 },
    { 0.15, 0.55, 0.5 },
    // Volume center
    { 0.65, 0.55, 0.5 } }
};

Kokkos::View<type_real ***, Kokkos::HostSpace>
compute_interpolant_polynomial_3d(
    const type_real xi_target, const type_real eta_target,
    const type_real zeta_target, const int N,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &xi) {

  auto [hxi, hpxi] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          xi_target, N, xi);
  auto [heta, hpeta] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          eta_target, N, xi);
  auto [hzeta, hpzeta] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          zeta_target, N, xi);

  Kokkos::View<type_real ***, Kokkos::HostSpace> polynomial("polynomial", N, N,
                                                            N);
  for (int iz = 0; iz < N; ++iz) {
    for (int iy = 0; iy < N; ++iy) {
      for (int ix = 0; ix < N; ++ix) {
        polynomial(iz, iy, ix) = hxi(ix) * heta(iy) * hzeta(iz);
      }
    }
  }

  return polynomial;
}

class SingleElementMesh3D {
public:
  SingleElementMesh3D() {}

  int ngnod; ///< Number of control nodes
  SingleElementMesh3D(int ngnod)
      : ngnod(ngnod), Coordinates("Coordinates", ngnod) {}

  template <typename Generator>
  SingleElementMesh3D(int ngnod, Generator generator)
      : ngnod(ngnod), Coordinates("Coordinates", ngnod) {
    for (int inode = 0; inode < ngnod; ++inode) {
      for (int idim = 0; idim < 3; ++idim) {
        Coordinates(inode, idim) = generator(inode, idim);
      }
    }
  }

  Kokkos::View<type_real *[3], Kokkos::HostSpace> Coordinates;

  std::tuple<type_real, type_real, type_real>
  get_coordinates(type_real xi, type_real eta, type_real zeta) const {
    const auto shape3D =
        specfem::shape_function::shape_function(xi, eta, zeta, ngnod);
    type_real xcor = 0.0;
    type_real ycor = 0.0;
    type_real zcor = 0.0;

    for (int in = 0; in < ngnod; in++) {
      xcor += Coordinates(in, 0) * shape3D[in];
      ycor += Coordinates(in, 1) * shape3D[in];
      zcor += Coordinates(in, 2) * shape3D[in];
    }

    return { xcor, ycor, zcor };
  }
};

class InterpolateFunction3D : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up any necessary data or state before each test
  }

  void TearDown() override {
    // Clean up any resources after each test
  }

  SingleElementMesh3D cube_8node_element;
  SingleElementMesh3D cube_27node_element;
  SingleElementMesh3D distorted_8node_element;
  SingleElementMesh3D distorted_27node_element;

  InterpolateFunction3D()
      : cube_8node_element(
            8,
            [](int ngnod, int dim) { return cube_8node_coords[ngnod][dim]; }),
        cube_27node_element(
            27,
            [](int ngnod, int dim) { return cube_27node_coords[ngnod][dim]; }),
        distorted_8node_element(8,
                                [](int ngnod, int dim) {
                                  return distorted_8node_coords[ngnod][dim];
                                }),
        distorted_27node_element(27, [](int ngnod, int dim) {
          return distorted_27node_coords[ngnod][dim];
        }) {}
};

TEST_F(InterpolateFunction3D, LinearFunction8NodeElement) {
  const int NGLL = 5;
  // Simple linear function
  const auto function = [](type_real x, type_real y, type_real z) {
    return x + y + z;
  };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at various points in the element
  const type_real xi_target = 0.5;
  const type_real eta_target = 0.5;
  const type_real zeta_target = 0.5;

  {
    auto [x, y, z] =
        cube_8node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_8node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in 8-node cube element";
  }

  {
    auto [x, y, z] = distorted_8node_element.get_coordinates(
        xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] = distorted_8node_element.get_coordinates(
              hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in distorted 8-node "
           "element";
  }
}

TEST_F(InterpolateFunction3D, LinearFunction27NodeElement) {
  const int NGLL = 5;
  // Simple linear function
  const auto function = [](type_real x, type_real y, type_real z) {
    return x + y + z;
  };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at off-center point
  const type_real xi_target = 0.25;
  const type_real eta_target = 0.25;
  const type_real zeta_target = 0.25;

  {
    auto [x, y, z] =
        cube_27node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_27node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in 27-node cube element";
  }

  {
    auto [x, y, z] = distorted_27node_element.get_coordinates(
        xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] = distorted_27node_element.get_coordinates(
              hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for linear function in distorted 27-node "
           "element";
  }
}

TEST_F(InterpolateFunction3D, QuadraticFunction8NodeElement) {
  const int NGLL = 5;
  // Simple quadratic function
  const auto function = [](type_real x, type_real y, type_real z) {
    return x * x + y * y + z * z;
  };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at the center of the element
  const type_real xi_target = 0.5;
  const type_real eta_target = 0.5;
  const type_real zeta_target = 0.5;

  {
    auto [x, y, z] =
        cube_8node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_8node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in 8-node cube element";
  }

  {
    auto [x, y, z] = distorted_8node_element.get_coordinates(
        xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] = distorted_8node_element.get_coordinates(
              hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in distorted 8-node "
           "element";
  }
}

TEST_F(InterpolateFunction3D, QuadraticFunction27NodeElement) {
  const int NGLL = 5;
  // Simple quadratic function
  const auto function = [](type_real x, type_real y, type_real z) {
    return x * x + y * y + z * z;
  };

  // Generate 5 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at various points
  const type_real xi_target = 0.75;
  const type_real eta_target = 0.25;
  const type_real zeta_target = 0.5;

  {
    auto [x, y, z] =
        cube_27node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_27node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in 27-node cube "
           "element";
  }

  {
    auto [x, y, z] = distorted_27node_element.get_coordinates(
        xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] = distorted_27node_element.get_coordinates(
              hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-6)
        << "Interpolation failed for quadratic function in distorted 27-node "
           "element";
  }
}

TEST_F(InterpolateFunction3D, CubicFunction8NodeElement) {
  const int NGLL = 6; // Higher order for cubic function
  // Cubic function
  const auto function = [](type_real x, type_real y, type_real z) {
    return x * x * x + y * y * y + z * z * z + x * y * z;
  };

  // Generate 6 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at various points
  const type_real xi_target = 0.3;
  const type_real eta_target = 0.7;
  const type_real zeta_target = 0.4;

  {
    auto [x, y, z] =
        cube_8node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_8node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-5)
        << "Interpolation failed for cubic function in 8-node cube element";
  }
}

TEST_F(InterpolateFunction3D, TrigonometricFunction8NodeElement) {
  const int NGLL = 7; // Higher order for smooth function
  // Trigonometric function
  const auto function = [](type_real x, type_real y, type_real z) {
    return std::sin(M_PI * x) * std::cos(M_PI * y) * std::sin(M_PI * z);
  };

  // Generate 7 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at center
  const type_real xi_target = 0.5;
  const type_real eta_target = 0.5;
  const type_real zeta_target = 0.5;

  {
    auto [x, y, z] =
        cube_8node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_8node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-4)
        << "Interpolation failed for trigonometric function in 8-node cube "
           "element";
  }
}

TEST_F(InterpolateFunction3D, ConstantFunction) {
  const int NGLL = 3; // Lower order sufficient for constant
  // Constant function
  const auto function = [](type_real x, type_real y, type_real z) {
    return 42.0;
  };

  // Generate 3 node quadrature
  specfem::quadrature::gll::gll gll(0.0, 0.0, NGLL);

  // Interpolate at any point
  const type_real xi_target = 0.8;
  const type_real eta_target = 0.2;
  const type_real zeta_target = 0.9;

  {
    auto [x, y, z] =
        cube_8node_element.get_coordinates(xi_target, eta_target, zeta_target);
    type_real function_value = function(x, y, z);

    // Create polynomial view
    const auto hxi = gll.get_hxi();
    const auto polynomial = compute_interpolant_polynomial_3d(
        xi_target, eta_target, zeta_target, NGLL, hxi);

    // Create function view
    Kokkos::View<type_real ***, Kokkos::HostSpace> function_view(
        "function", NGLL, NGLL, NGLL);
    for (int iz = 0; iz < NGLL; ++iz) {
      for (int iy = 0; iy < NGLL; ++iy) {
        for (int ix = 0; ix < NGLL; ++ix) {
          auto [x, y, z] =
              cube_8node_element.get_coordinates(hxi(ix), hxi(iy), hxi(iz));
          function_view(iz, iy, ix) = function(x, y, z);
        }
      }
    }

    // Interpolate the function
    auto function_interpolated =
        specfem::algorithms::interpolate_function(polynomial, function_view);

    EXPECT_NEAR(function_value, function_interpolated, 1e-12)
        << "Interpolation failed for constant function in 8-node cube element";
  }
}
