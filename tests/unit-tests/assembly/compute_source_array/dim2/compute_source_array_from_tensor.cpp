#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_tensor.hpp"
#include "../../test_fixture/test_fixture.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

// Helper function to test a tensor source with simplified jacobian (all
// derivatives = 1.0)
template <typename SourceType>
void test_tensor_source(const std::string &source_name, SourceType &source,
                        int ngll) {
  SCOPED_TRACE("Testing " + source_name);

  // Create quadrature::quadratures from GLL quadrature first
  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);

  // Create mesh_impl quadrature from quadratures object
  specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  // Get the source tensor for this source to determine number of components
  auto source_tensor = source.get_source_tensor();
  int ncomponents = source_tensor.extent(0);

  // Create source array for testing
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll);

  // Create simplified jacobian matrix with all derivatives set to 1.0
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                      false>;
  specfem::kokkos::HostView2d<PointJacobianMatrix> element_jacobian(
      "element_jacobian", ngll, ngll);

  // Set all jacobian derivatives to 1.0 for simplified testing
  // This means: dx/dxi = dx/dgamma = dz/dxi = dz/dgamma = 1.0
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Loop over all GLL points
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      SCOPED_TRACE("Testing GLL point (ix=" + std::to_string(ix) +
                   ", iz=" + std::to_string(iz) + ")");

      // Set source location to this GLL point
      const auto local_coords =
          specfem::point::local_coordinates<specfem::dimension::type::dim2>(
              0, xi_gamma_points(ix), xi_gamma_points(iz));
      source.set_local_coordinates(local_coords);

      // Initialize source array to zero
      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            source_array(ic, jz, jx) = 0.0;
          }
        }
      }

      // Compute source array using the testable helper function
      specfem::assembly::compute_source_array_impl::
          compute_source_array_from_tensor_and_element_jacobian(
              source, element_jacobian, quadrature, source_array);

      // For simplified jacobian (all derivatives = 1.0), we need to compute
      // expected derivatives properly First, compute the Lagrange interpolants
      // and their derivatives at the source location
      auto [hxi_source, hpxi_source] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              xi_gamma_points(ix), ngll, xi_gamma_points);
      auto [hgamma_source, hpgamma_source] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              xi_gamma_points(iz), ngll, xi_gamma_points);

      // Now compute derivatives at each GLL point
      for (int jz = 0; jz < ngll; ++jz) {
        for (int jx = 0; jx < ngll; ++jx) {
          // With simplified jacobian (all derivatives = 1.0):
          // dsrc_dx = hpxi_source(jx) * hgamma_source(jz) + hxi_source(jx) *
          // hpgamma_source(jz) dsrc_dz = hpxi_source(jx) * hgamma_source(jz) +
          // hxi_source(jx) * hpgamma_source(jz)
          type_real dsrc_dx = hpxi_source(jx) * hgamma_source(jz) +
                              hxi_source(jx) * hpgamma_source(jz);
          type_real dsrc_dz = hpxi_source(jx) * hgamma_source(jz) +
                              hxi_source(jx) * hpgamma_source(jz);

          // Note: for simplified jacobian, dsrc_dx = dsrc_dz

          // Verify source array matches expected tensor contraction
          for (int ic = 0; ic < ncomponents; ++ic) {
            type_real expected_value =
                source_tensor(ic, 0) * dsrc_dx + source_tensor(ic, 1) * dsrc_dz;

            EXPECT_NEAR(source_array(ic, jz, jx), expected_value, 1e-5)
                << "Component " << ic << " at GLL point (" << jx << "," << jz
                << ") should match expected tensor contraction when source is "
                   "at ("
                << ix << "," << iz << ")";
          }
        }
      }
    }
  }
}

// Helper function to test tensor source at off-GLL points where derivatives are
// non-zero
template <typename SourceType>
void test_tensor_source_off_gll(const std::string &source_name,
                                SourceType &source, int ngll) {
  SCOPED_TRACE("Testing " + source_name + " at off-GLL points");

  // Create quadrature::quadratures from GLL quadrature first
  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);

  // Create mesh_impl quadrature from quadratures object
  specfem::assembly::mesh_impl::quadrature<specfem::dimension::type::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  // Get the source tensor for this source to determine number of components
  auto source_tensor = source.get_source_tensor();
  int ncomponents = source_tensor.extent(0);

  // Create source array for testing
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll);

  // Create simplified jacobian matrix with all derivatives set to 1.0
  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                      false>;
  specfem::kokkos::HostView2d<PointJacobianMatrix> element_jacobian(
      "element_jacobian", ngll, ngll);

  // Set all jacobian derivatives to 1.0 for simplified testing
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Test at a few off-GLL points where derivatives will be non-zero
  std::vector<type_real> test_points = { -0.5, 0.0,
                                         0.5 }; // Points between GLL nodes

  for (type_real xi_source : test_points) {
    for (type_real gamma_source : test_points) {
      SCOPED_TRACE("Testing off-GLL point (xi=" + std::to_string(xi_source) +
                   ", gamma=" + std::to_string(gamma_source) + ")");

      // Set source location to this off-GLL point
      const auto local_coords =
          specfem::point::local_coordinates<specfem::dimension::type::dim2>(
              0, xi_source, gamma_source);
      source.set_local_coordinates(local_coords);

      // Initialize source array to zero
      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            source_array(ic, jz, jx) = 0.0;
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
              xi_source, ngll, xi_gamma_points);
      auto [hgamma_source, hpgamma_source] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              gamma_source, ngll, xi_gamma_points);

      // Compute derivatives at each GLL point
      for (int iz = 0; iz < ngll; ++iz) {
        for (int ix = 0; ix < ngll; ++ix) {
          // With simplified jacobian (all derivatives = 1.0):
          type_real dsrc_dx = hpxi_source(ix) * hgamma_source(iz) +
                              hxi_source(ix) * hpgamma_source(iz);
          type_real dsrc_dz = hpxi_source(ix) * hgamma_source(iz) +
                              hxi_source(ix) * hpgamma_source(iz);

          // Note: for simplified jacobian, dsrc_dx = dsrc_dz = (derivative sum)
          type_real expected_derivative = dsrc_dx; // Same as dsrc_dz

          // Verify source array matches expected tensor contraction
          for (int ic = 0; ic < ncomponents; ++ic) {
            type_real expected_value =
                source_tensor(ic, 0) * dsrc_dx + source_tensor(ic, 1) * dsrc_dz;

            // For our simplified jacobian: expected_value =
            // (source_tensor(ic,0) + source_tensor(ic,1)) * expected_derivative
            type_real simplified_expected =
                (source_tensor(ic, 0) + source_tensor(ic, 1)) *
                expected_derivative;

            EXPECT_NEAR(source_array(ic, iz, ix), simplified_expected, 1e-5)
                << "Component " << ic << " at GLL point (" << ix << "," << iz
                << ") should match expected tensor contraction";
          }
        }
      }
    }
  }
}

TEST(ASSEMBLY_NO_LOAD, compute_source_array_from_tensor) {

  const int ngll = 5;

  // Test Moment Tensor sources with different configurations

  // (1,0,0) - Mxx only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> moment_xx(
        0.0, 0.0,      // x, z
        1.0, 0.0, 0.0, // Mxx=1, Mzz=0, Mxz=0
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_xx.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxx (1,0,0)", moment_xx, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxx (1,0,0)", moment_xx, ngll);
  }

  // (0,1,0) - Mzz only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> moment_zz(
        0.0, 0.0,      // x, z
        0.0, 1.0, 0.0, // Mxx=0, Mzz=1, Mxz=0
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_zz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mzz (0,1,0)", moment_zz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mzz (0,1,0)", moment_zz, ngll);
  }

  // (0,0,1) - Mxz only
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> moment_xz(
        0.0, 0.0,      // x, z
        0.0, 0.0, 1.0, // Mxx=0, Mzz=0, Mxz=1
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_xz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxz (0,0,1)", moment_xz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxz (0,0,1)", moment_xz, ngll);
  }

  // (1,1,0) - Mxx and Mzz
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2>
        moment_xx_zz(0.0, 0.0,      // x, z
                     1.0, 1.0, 0.0, // Mxx=1, Mzz=1, Mxz=0
                     std::make_unique<specfem::forcing_function::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::wavefield::simulation_field::forward);
    moment_xx_zz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxx+Mzz (1,1,0)", moment_xx_zz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxx+Mzz (1,1,0)", moment_xx_zz,
                               ngll);
  }

  // (1,1,1) - All components
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> moment_all(
        0.0, 0.0,      // x, z
        1.0, 1.0, 1.0, // Mxx=1, Mzz=1, Mxz=1
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_all.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor All (1,1,1)", moment_all, ngll);
    test_tensor_source_off_gll("Moment Tensor All (1,1,1)", moment_all, ngll);
  }

  // (0,0,0) - Zero tensor
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> moment_zero(
        0.0, 0.0,      // x, z
        0.0, 0.0, 0.0, // Mxx=0, Mzz=0, Mxz=0
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        specfem::wavefield::simulation_field::forward);
    moment_zero.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Zero (0,0,0)", moment_zero, ngll);
    test_tensor_source_off_gll("Moment Tensor Zero (0,0,0)", moment_zero, ngll);
  }

  // Test with elastic_psv_t medium (3 components)
  {
    specfem::sources::moment_tensor<specfem::dimension::type::dim2>
        moment_psv_t(0.0, 0.0,      // x, z
                     1.0, 2.0, 0.5, // Mxx=1, Mzz=2, Mxz=0.5
                     std::make_unique<specfem::forcing_function::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::wavefield::simulation_field::forward);
    moment_psv_t.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_tensor_source("Moment Tensor PSV-T (1,2,0.5)", moment_psv_t, ngll);
    test_tensor_source_off_gll("Moment Tensor PSV-T (1,2,0.5)", moment_psv_t,
                               ngll);
  }
}
