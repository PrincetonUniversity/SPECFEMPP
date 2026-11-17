#include "specfem/assembly/compute_source_array/dim3/impl/compute_source_array_from_tensor.hpp"
#include "../../test_fixture/test_fixture.hpp"
#include "specfem/assembly/compute_source_array/dim3/impl/compute_source_array_from_vector.hpp"

#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

// Test-only tensor source with a configurable source tensor and body-couple
// vector, the 3D analog of the 2D BodyCoupleTestSource (issue #2113). Named
// (not anonymous) to stay unity-build safe.
class BodyCoupleTestSource3D : public specfem::sources::tensor_source<
                                   specfem::element::dimension_tag::dim3> {
public:
  BodyCoupleTestSource3D(std::vector<std::vector<type_real>> tensor,
                         std::vector<type_real> body_couple)
      : tensor_(std::move(tensor)), body_couple_(std::move(body_couple)) {}

  std::string source_name() const override { return "body-couple test source"; }

  specfem::simulation::field_type get_wavefield_type() const override {
    return specfem::simulation::field_type::forward;
  }

  std::vector<specfem::element::medium_tag>
  get_supported_media() const override {
    return { specfem::element::medium_tag::elastic_spin };
  }

  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_source_tensor() const override {
    const int nrows = static_cast<int>(tensor_.size());
    const int ncols = nrows > 0 ? static_cast<int>(tensor_[0].size()) : 0;
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_tensor("source_tensor", nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < ncols; ++j) {
        source_tensor(i, j) = tensor_[i][j];
      }
    }
    return source_tensor;
  }

  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_body_couple_vector() const override {
    if (body_couple_.empty()) {
      return {};
    }
    Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        body_couple("body_couple_vector", body_couple_.size());
    for (std::size_t c = 0; c < body_couple_.size(); ++c) {
      body_couple(c) = body_couple_[c];
    }
    return body_couple;
  }

  bool has_monopole_contribution() const override { return true; }

private:
  std::vector<std::vector<type_real>> tensor_;
  std::vector<type_real> body_couple_;
};

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
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
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
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
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
        const auto local_coords = specfem::point::local_coordinates<
            specfem::element::dimension_tag::dim3>(0, xi_eta_gamma_points(ix),
                                                   xi_eta_gamma_points(iy),
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
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
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
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
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
        const auto local_coords = specfem::point::local_coordinates<
            specfem::element::dimension_tag::dim3>(0, xi_source, eta_source,
                                                   gamma_source);
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
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_xx(0.0, 0.0, 0.0,                // x, y, z
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Mxx=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_xx.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxx (1,0,0,0,0,0)", moment_xx, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxx (1,0,0,0,0,0)", moment_xx,
                                  ngll);
  }

  // (0,1,0,0,0,0) - Myy only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_yy(0.0, 0.0, 0.0,                // x, y, z
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Myy=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_yy.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Myy (0,1,0,0,0,0)", moment_yy, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Myy (0,1,0,0,0,0)", moment_yy,
                                  ngll);
  }

  // (0,0,1,0,0,0) - Mzz only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_zz(0.0, 0.0, 0.0,                // x, y, z
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // Mzz=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_zz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mzz (0,0,1,0,0,0)", moment_zz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mzz (0,0,1,0,0,0)", moment_zz,
                                  ngll);
  }

  // (0,0,0,1,0,0) - Mxy only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_xy(0.0, 0.0, 0.0,                // x, y, z
                  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Mxy=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_xy.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxy (0,0,0,1,0,0)", moment_xy, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxy (0,0,0,1,0,0)", moment_xy,
                                  ngll);
  }

  // (0,0,0,0,1,0) - Mxz only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_xz(0.0, 0.0, 0.0,                // x, y, z
                  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // Mxz=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_xz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxz (0,0,0,0,1,0)", moment_xz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxz (0,0,0,0,1,0)", moment_xz,
                                  ngll);
  }

  // (0,0,0,0,0,1) - Myz only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_yz(0.0, 0.0, 0.0,                // x, y, z
                  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // Myz=1, others=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_yz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Myz (0,0,0,0,0,1)", moment_yz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Myz (0,0,0,0,0,1)", moment_yz,
                                  ngll);
  }

  // (1,1,1,0,0,0) - Mxx, Myy, and Mzz
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_xx_yy_zz(
            0.0, 0.0, 0.0,                // x, y, z
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0, // Mxx=Myy=Mzz=1,
                                          // off-diagonal=0
            std::make_unique<specfem::source_time_functions::Ricker>(
                10, 0.01, 1.0, 0.0, 1.0, false),
            specfem::simulation::field_type::forward);
    moment_xx_yy_zz.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mxx+Myy+Mzz (1,1,1,0,0,0)",
                          moment_xx_yy_zz, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mxx+Myy+Mzz (1,1,1,0,0,0)",
                                  moment_xx_yy_zz, ngll);
  }

  // (1,1,1,1,1,1) - All components
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_all(0.0, 0.0, 0.0,                // x, y, z
                   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // All components = 1
                   std::make_unique<specfem::source_time_functions::Ricker>(
                       10, 0.01, 1.0, 0.0, 1.0, false),
                   specfem::simulation::field_type::forward);
    moment_all.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor All (1,1,1,1,1,1)", moment_all, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor All (1,1,1,1,1,1)", moment_all,
                                  ngll);
  }

  // (0,0,0,0,0,0) - Zero tensor
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_zero(0.0, 0.0, 0.0,                // x, y, z
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // All components = 0
                    std::make_unique<specfem::source_time_functions::Ricker>(
                        10, 0.01, 1.0, 0.0, 1.0, false),
                    specfem::simulation::field_type::forward);
    moment_zero.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Zero (0,0,0,0,0,0)", moment_zero,
                          ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Zero (0,0,0,0,0,0)",
                                  moment_zero, ngll);
  }

  // Test with mixed non-zero values
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        moment_mixed(0.0, 0.0, 0.0,                // x, y, z
                     1.0, 2.0, 3.0, 0.5, 1.5, 2.5, // Mixed values
                     std::make_unique<specfem::source_time_functions::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::simulation::field_type::forward);
    moment_mixed.set_medium_tag(specfem::element::medium_tag::elastic);
    test_tensor_source_3d("Moment Tensor Mixed (1,2,3,0.5,1.5,2.5)",
                          moment_mixed, ngll);
    test_tensor_source_3d_off_gll("Moment Tensor Mixed (1,2,3,0.5,1.5,2.5)",
                                  moment_mixed, ngll);
  }
}

// Spike test: a pure body couple (0,0,0, cx,cy,cz) applied through the monopole
// path must reproduce, exactly, a cosserat_force with zero elastic force
// (f = 0) and rotational couple (fc_x, fc_y, fc_z) = (cx, cy, cz) applied
// through the vector (monopole) path. cosserat_force is the trusted reference
// for the monopole term (issue #2113, mirrors the 2D spike from #2111).
TEST(ASSEMBLY_NO_LOAD, moment_tensor_monopole_matches_cosserat_force_3d) {

  const int ngll = 5;
  const int ncomponents = 6;
  const type_real cx = 0.25;
  const type_real cy = -0.6;
  const type_real cz = 1.75;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // Zero tensor + body couple on the rotation rows: contributes only through
  // the monopole.
  BodyCoupleTestSource3D couple_source({ { 0.0, 0.0, 0.0 },
                                         { 0.0, 0.0, 0.0 },
                                         { 0.0, 0.0, 0.0 },
                                         { 0.0, 0.0, 0.0 },
                                         { 0.0, 0.0, 0.0 },
                                         { 0.0, 0.0, 0.0 } },
                                       { 0.0, 0.0, 0.0, cx, cy, cz });

  // Trusted reference: cosserat force with no elastic part (f = 0, fc =
  // couple).
  specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3>
      force_source(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cx, cy, cz,
                   std::make_unique<specfem::source_time_functions::Ricker>(
                       10, 0.01, 1.0, 0.0, 1.0, false),
                   specfem::simulation::field_type::forward);
  force_source.set_medium_tag(specfem::element::medium_tag::elastic_spin);

  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      couple_array("couple_array", ncomponents, ngll, ngll, ngll);
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      force_array("force_array", ncomponents, ngll, ngll, ngll);

  std::vector<type_real> test_points = { xi_eta_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_eta_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real eta_source : test_points) {
      for (type_real gamma_source : test_points) {
        SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                     ", eta=" + std::to_string(eta_source) +
                     ", gamma=" + std::to_string(gamma_source) + ")");
        const auto local_coords = specfem::point::local_coordinates<
            specfem::element::dimension_tag::dim3>(0, xi_source, eta_source,
                                                   gamma_source);
        couple_source.set_local_coordinates(local_coords);
        force_source.set_local_coordinates(local_coords);

        Kokkos::deep_copy(couple_array, 0.0);

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                couple_source, element_jacobian, quadrature, couple_array);
        specfem::assembly::compute_source_array_impl::
            accumulate_vector_contribution(
                couple_source.get_local_coordinates(),
                couple_source.get_body_couple_vector(), couple_array);
        specfem::assembly::compute_source_array_impl::from_vector(force_source,
                                                                  force_array);

        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                EXPECT_NEAR(couple_array(ic, jz, jy, jx),
                            force_array(ic, jz, jy, jx), 1e-5)
                    << "Monopole body couple must match cosserat_force at "
                    << "component " << ic << " point (" << jx << "," << jy
                    << "," << jz << ")";
              }
            }
          }
        }
      }
    }
  }
}

// Verify accumulate_vector_contribution adds the body-couple term
// L * body_couple(c) on top of the dipole contribution in 3D, at GLL and
// off-GLL points (issue #2113).
TEST(ASSEMBLY_NO_LOAD, accumulate_monopole_combines_with_dipole_3d) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // A 6-row tensor with generic entries plus a body couple on every row.
  BodyCoupleTestSource3D source({ { 0.3, -0.2, 0.1 },
                                  { 0.7, 1.1, -0.5 },
                                  { -0.4, 0.9, 0.2 },
                                  { 0.0, 0.0, 0.0 },
                                  { 0.0, 0.0, 0.0 },
                                  { 0.0, 0.0, 0.0 } },
                                { 0.0, 0.0, 0.0, 0.25, -0.6, 1.75 });

  const auto source_tensor = source.get_source_tensor();
  const auto body_couple = source.get_body_couple_vector();
  const int ncomponents = source_tensor.extent(0);

  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  std::vector<type_real> test_points = { xi_eta_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_eta_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real eta_source : test_points) {
      for (type_real gamma_source : test_points) {
        SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                     ", eta=" + std::to_string(eta_source) +
                     ", gamma=" + std::to_string(gamma_source) + ")");
        source.set_local_coordinates(specfem::point::local_coordinates<
                                     specfem::element::dimension_tag::dim3>(
            0, xi_source, eta_source, gamma_source));

        Kokkos::deep_copy(source_array, 0.0);

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);
        specfem::assembly::compute_source_array_impl::
            accumulate_vector_contribution(source.get_local_coordinates(),
                                           source.get_body_couple_vector(),
                                           source_array);

        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_source, ngll, xi_eta_gamma_points);
        auto [heta_source, hpeta_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                eta_source, ngll, xi_eta_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                gamma_source, ngll, xi_eta_gamma_points);

        for (int jz = 0; jz < ngll; ++jz) {
          for (int jy = 0; jy < ngll; ++jy) {
            for (int jx = 0; jx < ngll; ++jx) {
              // Simplified jacobian (all derivatives 1.0) => the three spatial
              // derivatives are equal.
              const type_real dsrc =
                  hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);
              const type_real hlagrange =
                  hxi_source(jx) * heta_source(jy) * hgamma_source(jz);
              for (int ic = 0; ic < ncomponents; ++ic) {
                const type_real expected =
                    source_tensor(ic, 0) * dsrc + source_tensor(ic, 1) * dsrc +
                    source_tensor(ic, 2) * dsrc + hlagrange * body_couple(ic);
                EXPECT_NEAR(source_array(ic, jz, jy, jx), expected, 1e-5)
                    << "Component " << ic << " at GLL point (" << jx << ","
                    << jy << "," << jz << ")";
              }
            }
          }
        }
      }
    }
  }
}

// A non-empty, all-zero body couple must not change the dipole result in 3D
// (issue #2113).
TEST(ASSEMBLY_NO_LOAD, accumulate_monopole_zero_body_couple_adds_nothing_3d) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  BodyCoupleTestSource3D source({ { 1.0, 0.5, 0.2 },
                                  { 0.5, 2.0, 0.1 },
                                  { 0.2, 0.1, 1.5 },
                                  { 0.0, 0.0, 0.0 },
                                  { 0.0, 0.0, 0.0 },
                                  { 0.0, 0.0, 0.0 } },
                                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 6u);

  const int ncomponents = source.get_source_tensor().extent(0);

  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      dipole_only("dipole_only", ncomponents, ngll, ngll, ngll);
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      with_monopole("with_monopole", ncomponents, ngll, ngll, ngll);

  std::vector<type_real> test_points = { xi_eta_gamma_points(0), 0.0,
                                         xi_eta_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real eta_source : test_points) {
      for (type_real gamma_source : test_points) {
        source.set_local_coordinates(specfem::point::local_coordinates<
                                     specfem::element::dimension_tag::dim3>(
            0, xi_source, eta_source, gamma_source));

        Kokkos::deep_copy(dipole_only, 0.0);
        Kokkos::deep_copy(with_monopole, 0.0);

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, dipole_only);
        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, with_monopole);
        specfem::assembly::compute_source_array_impl::
            accumulate_vector_contribution(source.get_local_coordinates(),
                                           source.get_body_couple_vector(),
                                           with_monopole);

        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                EXPECT_NEAR(with_monopole(ic, jz, jy, jx),
                            dipole_only(ic, jz, jy, jx), 1e-12)
                    << "Zero body couple must not change the dipole result at "
                    << "component " << ic << " point (" << jx << "," << jy
                    << "," << jz << ")";
              }
            }
          }
        }
      }
    }
  }
}

// An asymmetric moment tensor on elastic_spin drives both the displacement rows
// (dipole M*grad L) and the rotation rows (monopole L*(eps:M)). Verifies the
// full combined result, confirms elastic_spin no longer aborts, and pins the
// couple sign convention (issue #2113).
TEST(ASSEMBLY_NO_LOAD, asymmetric_moment_tensor_dipole_plus_monopole_3d) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // (Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy). The second case is
  // antisymmetric in a single pair at a time, to pin each couple component.
  struct Params {
    type_real Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy;
  };
  std::vector<Params> cases = {
    { 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7 },
    { 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, -0.3, 0.0, 0.0 }, // only (eps:M)_z =
                                                      // Mxy-Myx
    { 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.4, 0.0 }, // only (eps:M)_y =
                                                      // Mzx-Mxz
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5 } // only (eps:M)_x = Myz-Mzy
  };

  for (const auto &p : cases) {
    SCOPED_TRACE("asymmetric case");
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
        source(0.0, 0.0, 0.0, p.Mxx, p.Myy, p.Mzz, p.Mxy, p.Mxz, p.Myz, p.Myx,
               p.Mzx, p.Mzy,
               std::make_unique<specfem::source_time_functions::Ricker>(
                   10, 0.01, 1.0, 0.0, 1.0, false),
               specfem::simulation::field_type::forward);
    source.set_medium_tag(specfem::element::medium_tag::elastic_spin);

    const auto source_tensor = source.get_source_tensor();
    const int ncomponents = source_tensor.extent(0);
    ASSERT_EQ(ncomponents, 6);

    // Expected body couple, by the issue's sign convention.
    const type_real couple_x = p.Myz - p.Mzy;
    const type_real couple_y = p.Mzx - p.Mxz;
    const type_real couple_z = p.Mxy - p.Myx;

    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array("source_array", ncomponents, ngll, ngll, ngll);

    std::vector<type_real> test_points = { xi_eta_gamma_points(0), 0.0, 0.5,
                                           xi_eta_gamma_points(ngll - 1) };

    for (type_real xi_source : test_points) {
      for (type_real eta_source : test_points) {
        for (type_real gamma_source : test_points) {
          source.set_local_coordinates(specfem::point::local_coordinates<
                                       specfem::element::dimension_tag::dim3>(
              0, xi_source, eta_source, gamma_source));

          Kokkos::deep_copy(source_array, 0.0);

          specfem::assembly::compute_source_array_impl::
              compute_source_array_from_tensor_and_element_jacobian(
                  source, element_jacobian, quadrature, source_array);
          specfem::assembly::compute_source_array_impl::
              accumulate_vector_contribution(source.get_local_coordinates(),
                                             source.get_body_couple_vector(),
                                             source_array);

          auto [hxi_source, hpxi_source] =
              specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                  xi_source, ngll, xi_eta_gamma_points);
          auto [heta_source, hpeta_source] =
              specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                  eta_source, ngll, xi_eta_gamma_points);
          auto [hgamma_source, hpgamma_source] =
              specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                  gamma_source, ngll, xi_eta_gamma_points);

          for (int jz = 0; jz < ngll; ++jz) {
            for (int jy = 0; jy < ngll; ++jy) {
              for (int jx = 0; jx < ngll; ++jx) {
                const type_real dsrc =
                    hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                    hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                    hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);
                const type_real hlagrange =
                    hxi_source(jx) * heta_source(jy) * hgamma_source(jz);

                // Displacement rows 0-2: dipole only.
                for (int ic = 0; ic < 3; ++ic) {
                  const type_real expected = source_tensor(ic, 0) * dsrc +
                                             source_tensor(ic, 1) * dsrc +
                                             source_tensor(ic, 2) * dsrc;
                  EXPECT_NEAR(source_array(ic, jz, jy, jx), expected, 1e-5);
                }
                // Rotation rows 3-5: monopole body couple only (the tensor's
                // rotation rows are zero).
                EXPECT_NEAR(source_array(3, jz, jy, jx), hlagrange * couple_x,
                            1e-5);
                EXPECT_NEAR(source_array(4, jz, jy, jx), hlagrange * couple_y,
                            1e-5);
                EXPECT_NEAR(source_array(5, jz, jy, jx), hlagrange * couple_z,
                            1e-5);
              }
            }
          }
        }
      }
    }
  }
}

// Regression: a symmetric moment tensor produces zero rotational coupling
// through the full tensor path on elastic_spin (issue #2113).
TEST(ASSEMBLY_NO_LOAD,
     symmetric_moment_tensor_has_zero_rotational_coupling_3d) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  // Symmetric tensor (9-arg ctor sets the lower triangle equal to the upper).
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3> source(
      0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      specfem::simulation::field_type::forward);
  source.set_medium_tag(specfem::element::medium_tag::elastic_spin);

  auto body_couple = source.get_body_couple_vector();
  ASSERT_EQ(body_couple.extent(0), 6u);
  EXPECT_NEAR(body_couple(3), 0.0, 1e-6);
  EXPECT_NEAR(body_couple(4), 0.0, 1e-6);
  EXPECT_NEAR(body_couple(5), 0.0, 1e-6);

  const int ncomponents = source.get_source_tensor().extent(0);
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  source.set_local_coordinates(
      specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>(
          0, 0.3, -0.4, 0.2));
  Kokkos::deep_copy(source_array, 0.0);

  specfem::assembly::compute_source_array_impl::
      compute_source_array_from_tensor_and_element_jacobian(
          source, element_jacobian, quadrature, source_array);
  specfem::assembly::compute_source_array_impl::accumulate_vector_contribution(
      source.get_local_coordinates(), source.get_body_couple_vector(),
      source_array);

  // Rotation rows must be identically zero.
  for (int ic = 3; ic < 6; ++ic) {
    for (int jz = 0; jz < ngll; ++jz) {
      for (int jy = 0; jy < ngll; ++jy) {
        for (int jx = 0; jx < ngll; ++jx) {
          EXPECT_NEAR(source_array(ic, jz, jy, jx), 0.0, 1e-12);
        }
      }
    }
  }
}

// A spin tensor populates only the three rotation rows of the source array,
// with the gradient contraction Mc*grad L; the three displacement rows stay
// identically zero and no monopole term is added (issue #2113).
TEST(ASSEMBLY_NO_LOAD, spin_tensor_fills_only_rotation_rows_3d) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim3>
      quadrature(quadratures);
  auto xi_eta_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        element_jacobian(iz, iy, ix) =
            PointJacobianMatrix(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
      }
    }
  }

  specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3> source(
      0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      specfem::simulation::field_type::forward);
  source.set_medium_tag(specfem::element::medium_tag::elastic_spin);

  // The spin tensor has no monopole contribution by design.
  EXPECT_FALSE(source.has_monopole_contribution());
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);

  // Exercise the generic contraction helpers (verifies the full 6x3 tensor
  // contraction at GLL and off-GLL points).
  test_tensor_source_3d("Spin Tensor 3D", source, ngll);
  test_tensor_source_3d_off_gll("Spin Tensor 3D", source, ngll);

  const auto source_tensor = source.get_source_tensor();
  const int ncomponents = source_tensor.extent(0);
  ASSERT_EQ(ncomponents, 6);
  Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll, ngll);

  std::vector<type_real> test_points = { xi_eta_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_eta_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real eta_source : test_points) {
      for (type_real gamma_source : test_points) {
        source.set_local_coordinates(specfem::point::local_coordinates<
                                     specfem::element::dimension_tag::dim3>(
            0, xi_source, eta_source, gamma_source));

        Kokkos::deep_copy(source_array, 0.0);

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);

        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_source, ngll, xi_eta_gamma_points);
        auto [heta_source, hpeta_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                eta_source, ngll, xi_eta_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                gamma_source, ngll, xi_eta_gamma_points);

        for (int jz = 0; jz < ngll; ++jz) {
          for (int jy = 0; jy < ngll; ++jy) {
            for (int jx = 0; jx < ngll; ++jx) {
              const type_real dsrc =
                  hpxi_source(jx) * heta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * hpeta_source(jy) * hgamma_source(jz) +
                  hxi_source(jx) * heta_source(jy) * hpgamma_source(jz);

              // Displacement rows 0-2 are identically zero.
              EXPECT_NEAR(source_array(0, jz, jy, jx), 0.0, 1e-12);
              EXPECT_NEAR(source_array(1, jz, jy, jx), 0.0, 1e-12);
              EXPECT_NEAR(source_array(2, jz, jy, jx), 0.0, 1e-12);
              // Rotation rows 3-5 carry the Mc contraction.
              for (int ic = 3; ic < 6; ++ic) {
                const type_real expected = source_tensor(ic, 0) * dsrc +
                                           source_tensor(ic, 1) * dsrc +
                                           source_tensor(ic, 2) * dsrc;
                EXPECT_NEAR(source_array(ic, jz, jy, jx), expected, 1e-5);
              }
            }
          }
        }
      }
    }
  }
}
