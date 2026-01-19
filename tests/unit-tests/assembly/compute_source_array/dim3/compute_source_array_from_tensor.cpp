#include "specfem/assembly/compute_source_array/dim3/impl/compute_source_array_from_tensor.hpp"
#include "../../test_fixture/test_fixture.hpp"
#include "kokkos_abstractions.h"
#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

// Helper function to test a tensor source with simplified jacobian (all
// derivatives = 1.0)
template <typename SourceType>
void test_tensor_source_3d(const std::string &source_name, SourceType &source,
                           int ngll) {
  SCOPED_TRACE("Testing " + source_name);

  // Create quadrature::quadratures from GLL quadrature first
  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);

  // Create mesh_impl quadrature from quadratures object
  specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  // Get the source tensor for this source to determine number of components
  auto source_tensor = source.get_source_tensor();
  int ncomponents = source_tensor.extent(0);

  // Create source array for testing (4D: [components, z, y, x])
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  // Create simplified jacobian matrix with all derivatives set to 1.0
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim3, false,
                                      false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);

  // Set all jacobian derivatives to 1.0 for simplified testing
  // This means: dx/dxi = dx/deta = dx/dgamma = dy/dxi = dy/deta = dy/dgamma =
  // dz/dxi = dz/deta = dz/dgamma = 1.0
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // Loop over all GLL points
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        SCOPED_TRACE("Testing GLL point (ix=" + std::to_string(ix) + ", iy=" +
                     std::to_string(iy) + ", iz=" + std::to_string(iz) + ")");

        // Set source location to this GLL point
        const auto local_coords =
            specfem::point::local_coordinates<specfem::dimension::type::dim3>(
                0, xi_eta_gamma_points(ix), xi_eta_gamma_points(iy),
                xi_eta_gamma_points(iz));
        source.set_local_coordinates(local_coords);

        // Initialize source array to zero
        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                source_array(ic, jz, jy, jx) = 0.0;
              }
            }
          }
        }

        // Compute source array using the testable helper function
        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);

        // For simplified jacobian (all derivatives = 1.0), we need to compute
        // expected derivatives properly First, compute the Lagrange
        // interpolants and their derivatives at the source location
        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_eta_gamma_points(ix), ngll, xi_eta_gamma_points);
        auto [heta_source, hpeta_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_eta_gamma_points(iy), ngll, xi_eta_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_eta_gamma_points(iz), ngll, xi_eta_gamma_points);

        // Now compute derivatives at each GLL point
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jy = 0; jy < ngll; ++jy) {
            for (int jx = 0; jx < ngll; ++jx) {
              // With simplified jacobian (all derivatives = 1.0):
              // dsrc_dx = hpxi_source(jx) * heta_source(jy) *
              // hgamma_source(jz) +
              //           hxi_source(jx) * hpeta_source(jy) *
              //           hgamma_source(jz) + hxi_source(jx) * heta_source(jy)
              //           * hpgamma_source(jz)
              // Same pattern for dsrc_dy and dsrc_dz
              type_real dsrc_dx =
                  hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);
              type_real dsrc_dy =
                  hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);
              type_real dsrc_dz =
                  hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);

              // Note: for simplified jacobian, dsrc_dx = dsrc_dy = dsrc_dz

              // Verify source array matches expected tensor contraction
              for (int ic = 0; ic < ncomponents; ++ic) {
                type_real expected_value = source_tensor(ic, 0) * dsrc_dx +
                                           source_tensor(ic, 1) * dsrc_dy +
                                           source_tensor(ic, 2) * dsrc_dz;

                EXPECT_NEAR(source_array(ic, jz, jy, jx), expected_value, 1e-5)
                    << "Component " << ic << " at GLL point (" << jx << ","
                    << jy << "," << jz
                    << ") should match expected tensor contraction when source "
                       "is "
                       "at ("
                    << ix << "," << iy << "," << iz << ")";
              }
            }
          }
        }
      }
    }
  }
}

// Helper function to test tensor source at off-GLL points where derivatives
// are non-zero
template <typename SourceType>
void test_tensor_source_3d_off_gll(const std::string &source_name,
                                   SourceType &source, int ngll) {
  SCOPED_TRACE("Testing " + source_name + " at off-GLL points");

  // Create quadrature::quadratures from GLL quadrature first
  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);

  // Create mesh_impl quadrature from quadratures object
  specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  // Get the source tensor for this source to determine number of components
  auto source_tensor = source.get_source_tensor();
  int ncomponents = source_tensor.extent(0);

  // Create source array for testing
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  // Create simplified jacobian matrix with all derivatives set to 1.0
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim3, false,
                                      false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);

  // Set all jacobian derivatives to 1.0 for simplified testing
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // Test at a few off-GLL points where derivatives will be non-zero
  std::vector<type_real> test_points = { -0.5, 0.0,
                                         0.5 }; // Points between GLL nodes

  for (type_real xi_source : test_points) {
    for (type_real eta_source : test_points) {
      for (type_real gamma_source : test_points) {
        SCOPED_TRACE("Testing off-GLL point (xi=" + std::to_string(xi_source) +
                     ", eta=" + std::to_string(eta_source) +
                     ", gamma=" + std::to_string(gamma_source) + ")");

        // Set source location to this off-GLL point
        const auto local_coords =
            specfem::point::local_coordinates<specfem::dimension::type::dim3>(
                0, xi_source, eta_source, gamma_source);
        source.set_local_coordinates(local_coords);

        // Initialize source array to zero
        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                source_array(ic, jz, jy, jx) = 0.0;
              }
            }
          }
        }

        // Compute source array using the testable helper function
        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);

        // Now manually compute expected derivatives for verification
        // Compute lagrange interpolants at the source location
        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_source, ngll, xi_eta_gamma_points);
        auto [heta_source, hpeta_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                eta_source, ngll, xi_eta_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                gamma_source, ngll, xi_eta_gamma_points);

        // Compute derivatives at each GLL point
        for (int iz = 0; iz < ngll; ++iz) {
          for (int iy = 0; iy < ngll; ++iy) {
            for (int ix = 0; ix < ngll; ++ix) {
              // With simplified jacobian (all derivatives = 1.0):
              type_real dsrc_dx =
                  hpxi_source(ix) * heta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * hpeta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * heta_source(iy) * hpgamma_source(iz);
              type_real dsrc_dy =
                  hpxi_source(ix) * heta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * hpeta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * heta_source(iy) * hpgamma_source(iz);
              type_real dsrc_dz =
                  hpxi_source(ix) * heta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * hpeta_source(iy) * hgamma_source(iz) +
                  hxi_source(ix) * heta_source(iy) * hpgamma_source(iz);

              // Note: for simplified jacobian, dsrc_dx = dsrc_dy = dsrc_dz =
              // (derivative sum)
              type_real expected_derivative = dsrc_dx; // Same as dsrc_dy and
                                                       // dsrc_dz

              // Verify source array matches expected tensor contraction
              for (int ic = 0; ic < ncomponents; ++ic) {
                type_real expected_value = source_tensor(ic, 0) * dsrc_dx +
                                           source_tensor(ic, 1) * dsrc_dy +
                                           source_tensor(ic, 2) * dsrc_dz;

                // For our simplified jacobian: expected_value =
                // (source_tensor(ic,0) + source_tensor(ic,1) +
                // source_tensor(ic,2)) * expected_derivative
                type_real simplified_expected =
                    (source_tensor(ic, 0) + source_tensor(ic, 1) +
                     source_tensor(ic, 2)) *
                    expected_derivative;

                EXPECT_NEAR(source_array(ic, iz, iy, ix), simplified_expected,
                            1e-5)
                    << "Component " << ic << " at GLL point (" << ix << ","
                    << iy << "," << iz
                    << ") should match expected tensor contraction";
              }
            }
          }
        }
      }
    }
  }
}

TEST(ASSEMBLY_NO_LOAD, compute_source_array_from_tensor_3d) {

  const int ngll = 5;

  // Test Moment Tensor sources with different configurations

  // (1,0,0,0,0,0) - Mxx only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_xx(
        0.0, 0.0, 0.0,                // x, y, z
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Mxx=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_xx.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxx (1,0,0,0,0,0)", moment_xx, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxx (1,0,0,0,0,0)", moment_xx,
                                  ngll);
  }

  // (0,1,0,0,0,0) - Myy only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_yy(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Myy=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_yy.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Myy (0,1,0,0,0,0)", moment_yy, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Myy (0,1,0,0,0,0)", moment_yy,
                                  ngll);
  }

  // (0,0,1,0,0,0) - Mzz only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_zz(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Mzz=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_zz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mzz (0,0,1,0,0,0)", moment_zz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mzz (0,0,1,0,0,0)", moment_zz,
                                  ngll);
  }

  // (0,0,0,1,0,0) - Mxy only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_xy(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Mxy=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_xy.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxy (0,0,0,1,0,0)", moment_xy, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxy (0,0,0,1,0,0)", moment_xy,
                                  ngll);
  }

  // (0,0,0,0,1,0) - Mxz only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_xz(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // Mxz=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_xz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxz (0,0,0,0,1,0)", moment_xz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxz (0,0,0,0,1,0)", moment_xz,
                                  ngll);
  }

  // (0,0,0,0,0,1) - Myz only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_yz(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // Myz=1, others=0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_yz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Myz (0,0,0,0,0,1)", moment_yz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Myz (0,0,0,0,0,1)", moment_yz,
                                  ngll);
  }

  // (1,1,1,0,0,0) - Mxx, Myy, and Mzz
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3>
        moment_xx_yy_zz(
            0.0, 0.0, 0.0,                // x, y, z
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, // Mxx=Myy=Mzz=1,
                                          // off-diagonal=0
            std::make_unique<specfem::source_time_functions::Ricker>(
                10, 0.01, 1.0, 0.0, 1.0, false),
            specfem::wavefield::simulation_field::forward);
    moment_xx_yy_zz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxx+Myy+Mzz (1,1,1,0,0,0)",
                          moment_xx_yy_zz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxx+Myy+Mzz (1,1,1,0,0,0)",
                                  moment_xx_yy_zz, ngll);
  }

  // (1,1,1,1,1,1) - All components
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_all(
        0.0, 0.0, 0.0,                // x, y, z
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // All components = 1
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_all.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor All (1,1,1,1,1,1)", moment_all, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor All (1,1,1,1,1,1)", moment_all,
                                  ngll);
  }

  // (0,0,0,0,0,0) - Zero tensor
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> moment_zero(
        0.0, 0.0, 0.0,                // x, y, z
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // All components = 0
        std::make_unique<specfem::source_time_functions::Ricker>(
            10, 0.01, 1.0, 0.0, 1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_zero.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Zero (0,0,0,0,0,0)", moment_zero,
                          ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Zero (0,0,0,0,0,0)",
                                  moment_zero, ngll);
  }

  // Test with mixed non-zero values
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim3>
        moment_mixed(0.0, 0.0, 0.0,                // x, y, z
                     1.0, 2.0, 3.0, 0.5, 1.5, 2.5, // Mixed values
                     std::make_unique<specfem::source_time_functions::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::wavefield::simulation_field::forward);
    moment_mixed.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mixed (1,2,3,0.5,1.5,2.5)",
                          moment_mixed, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mixed (1,2,3,0.5,1.5,2.5)",
                                  moment_mixed, ngll);
  }
}
