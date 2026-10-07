#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_tensor.hpp"
#include "../../test_fixture/test_fixture.hpp"
#include "specfem/assembly/compute_source_array/dim2/impl/compute_source_array_from_vector.hpp"

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
// vector (issue #2111). Named (not anonymous) to stay unity-build safe.
class BodyCoupleTestSource : public specfem::sources::tensor_source<
                                 specfem::element::dimension_tag::dim2> {
public:
  BodyCoupleTestSource(std::vector<std::vector<type_real>> tensor,
                       std::vector<type_real> body_couple)
      : tensor_(std::move(tensor)), body_couple_(std::move(body_couple)) {}

  std::string source_name() const override { return "body-couple test source"; }

  specfem::simulation::field_type get_wavefield_type() const override {
    return specfem::simulation::field_type::forward;
  }

  std::vector<specfem::element::medium_tag>
  get_supported_media() const override {
    return { specfem::element::medium_tag::elastic_psv_t };
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
void test_tensor_source(const std::string &source_name, SourceType &source,
                        int ngll) {
  SCOPED_TRACE("Testing " + source_name);

  // Create quadrature::quadratures from GLL quadrature first
  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);

  // Create mesh_impl quadrature from quadratures object
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
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
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);

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
      const auto local_coords = specfem::point::local_coordinates<
          specfem::element::dimension_tag::dim2>(0, xi_gamma_points(ix),
                                                 xi_gamma_points(iz));
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
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
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
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);

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
      const auto local_coords = specfem::point::local_coordinates<
          specfem::element::dimension_tag::dim2>(0, xi_source, gamma_source);
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
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_xx(0.0, 0.0,      // x, z
                  1.0, 0.0, 0.0, // Mxx=1, Mzz=0, Mxz=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_xx.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxx (1,0,0)", moment_xx, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxx (1,0,0)", moment_xx, ngll);
  }

  // (0,1,0) - Mzz only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_zz(0.0, 0.0,      // x, z
                  0.0, 1.0, 0.0, // Mxx=0, Mzz=1, Mxz=0
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_zz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mzz (0,1,0)", moment_zz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mzz (0,1,0)", moment_zz, ngll);
  }

  // (0,0,1) - Mxz only
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_xz(0.0, 0.0,      // x, z
                  0.0, 0.0, 1.0, // Mxx=0, Mzz=0, Mxz=1
                  std::make_unique<specfem::source_time_functions::Ricker>(
                      10, 0.01, 1.0, 0.0, 1.0, false),
                  specfem::simulation::field_type::forward);
    moment_xz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxz (0,0,1)", moment_xz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxz (0,0,1)", moment_xz, ngll);
  }

  // (1,1,0) - Mxx and Mzz
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_xx_zz(0.0, 0.0,      // x, z
                     1.0, 1.0, 0.0, // Mxx=1, Mzz=1, Mxz=0
                     std::make_unique<specfem::source_time_functions::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::simulation::field_type::forward);
    moment_xx_zz.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Mxx+Mzz (1,1,0)", moment_xx_zz, ngll);
    test_tensor_source_off_gll("Moment Tensor Mxx+Mzz (1,1,0)", moment_xx_zz,
                               ngll);
  }

  // (1,1,1) - All components
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_all(0.0, 0.0,      // x, z
                   1.0, 1.0, 1.0, // Mxx=1, Mzz=1, Mxz=1
                   std::make_unique<specfem::source_time_functions::Ricker>(
                       10, 0.01, 1.0, 0.0, 1.0, false),
                   specfem::simulation::field_type::forward);
    moment_all.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor All (1,1,1)", moment_all, ngll);
    test_tensor_source_off_gll("Moment Tensor All (1,1,1)", moment_all, ngll);
  }

  // (0,0,0) - Zero tensor
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_zero(0.0, 0.0,      // x, z
                    0.0, 0.0, 0.0, // Mxx=0, Mzz=0, Mxz=0
                    std::make_unique<specfem::source_time_functions::Ricker>(
                        10, 0.01, 1.0, 0.0, 1.0, false),
                    specfem::simulation::field_type::forward);
    moment_zero.set_medium_tag(specfem::element::medium_tag::elastic_psv);
    test_tensor_source("Moment Tensor Zero (0,0,0)", moment_zero, ngll);
    test_tensor_source_off_gll("Moment Tensor Zero (0,0,0)", moment_zero, ngll);
  }

  // Test with elastic_psv_t medium (3 components)
  {
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        moment_psv_t(0.0, 0.0,      // x, z
                     1.0, 2.0, 0.5, // Mxx=1, Mzz=2, Mxz=0.5
                     std::make_unique<specfem::source_time_functions::Ricker>(
                         10, 0.01, 1.0, 0.0, 1.0, false),
                     specfem::simulation::field_type::forward);
    moment_psv_t.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
    test_tensor_source("Moment Tensor PSV-T (1,2,0.5)", moment_psv_t, ngll);
    test_tensor_source_off_gll("Moment Tensor PSV-T (1,2,0.5)", moment_psv_t,
                               ngll);
  }
}

// A spin tensor populates only the rotation row of the source array, with the
// gradient contraction Mcyx * dL/dx + Mcyz * dL/dz; the displacement rows stay
// identically zero and no monopole term is added (issue #2112).
TEST(ASSEMBLY_NO_LOAD, spin_tensor_fills_only_rotation_row) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  const type_real Mcyx = 0.8;
  const type_real Mcyz = -0.3;
  specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2> source(
      0.0, 0.0, Mcyx, Mcyz,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      specfem::simulation::field_type::forward);
  source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);

  // The spin tensor has no monopole contribution by design.
  EXPECT_FALSE(source.has_monopole_contribution());
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);

  // Exercise the generic contraction helpers as well (verifies the full 3x2
  // tensor contraction at GLL and off-GLL points).
  test_tensor_source("Spin Tensor (Mcyx=0.8, Mcyz=-0.3)", source, ngll);
  test_tensor_source_off_gll("Spin Tensor (Mcyx=0.8, Mcyz=-0.3)", source, ngll);

  const int ncomponents = source.get_source_tensor().extent(0);
  ASSERT_EQ(ncomponents, 3);
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll);

  // On-GLL and off-GLL source positions.
  std::vector<type_real> test_points = { xi_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real gamma_source : test_points) {
      SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                   ", gamma=" + std::to_string(gamma_source) + ")");
      source.set_local_coordinates(specfem::point::local_coordinates<
                                   specfem::element::dimension_tag::dim2>(
          0, xi_source, gamma_source));

      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            source_array(ic, jz, jx) = 0.0;
          }
        }
      }

      specfem::assembly::compute_source_array_impl::
          compute_source_array_from_tensor_and_element_jacobian(
              source, element_jacobian, quadrature, source_array);

      auto [hxi_source, hpxi_source] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              xi_source, ngll, xi_gamma_points);
      auto [hgamma_source, hpgamma_source] =
          specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
              gamma_source, ngll, xi_gamma_points);

      for (int jz = 0; jz < ngll; ++jz) {
        for (int jx = 0; jx < ngll; ++jx) {
          // Simplified jacobian (all derivatives 1.0) => dsrc_dx == dsrc_dz.
          const type_real dsrc = hpxi_source(jx) * hgamma_source(jz) +
                                 hxi_source(jx) * hpgamma_source(jz);

          // Displacement rows are identically zero.
          EXPECT_NEAR(source_array(0, jz, jx), 0.0, 1e-12);
          EXPECT_NEAR(source_array(1, jz, jx), 0.0, 1e-12);
          // Rotation row carries the Mc contraction.
          EXPECT_NEAR(source_array(2, jz, jx), (Mcyx + Mcyz) * dsrc, 1e-5);
        }
      }
    }
  }
}

// Verify accumulate_vector_contribution adds the body-couple term
// L * body_couple(c) on top of the dipole contribution computed by the inner
// helper (issue #2111).
TEST(ASSEMBLY_NO_LOAD, accumulate_monopole_combines_with_dipole) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Two configurations: a Cosserat-like body couple on the rotation row, and a
  // fully generic couple with non-zero entries in every slot.
  std::vector<
      std::pair<std::vector<std::vector<type_real>>, std::vector<type_real>>>
      configs = {
        { { { 1.0, 0.5 }, { 0.5, 2.0 }, { 0.0, 0.0 } }, { 0.0, 0.0, 1.5 } },
        { { { 0.3, -0.2 }, { 0.7, 1.1 }, { -0.4, 0.9 } }, { 0.25, -0.6, 1.75 } }
      };

  // Source positions: on-GLL nodes and off-GLL points.
  std::vector<type_real> test_points = { xi_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_gamma_points(ngll - 1) };

  for (std::size_t cfg = 0; cfg < configs.size(); ++cfg) {
    SCOPED_TRACE("config " + std::to_string(cfg));
    BodyCoupleTestSource source(configs[cfg].first, configs[cfg].second);
    const auto source_tensor = source.get_source_tensor();
    const auto body_couple = source.get_body_couple_vector();
    const int ncomponents = source_tensor.extent(0);

    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array("source_array", ncomponents, ngll, ngll);

    for (type_real xi_source : test_points) {
      for (type_real gamma_source : test_points) {
        SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                     ", gamma=" + std::to_string(gamma_source) + ")");
        source.set_local_coordinates(specfem::point::local_coordinates<
                                     specfem::element::dimension_tag::dim2>(
            0, xi_source, gamma_source));

        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jx = 0; jx < ngll; ++jx) {
              source_array(ic, jz, jx) = 0.0;
            }
          }
        }

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);
        specfem::assembly::compute_source_array_impl::
            accumulate_vector_contribution(source.get_local_coordinates(),
                                           source.get_body_couple_vector(),
                                           source_array);

        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_source, ngll, xi_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                gamma_source, ngll, xi_gamma_points);

        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            // Simplified jacobian (all derivatives 1.0) => dsrc_dx == dsrc_dz.
            const type_real dsrc = hpxi_source(jx) * hgamma_source(jz) +
                                   hxi_source(jx) * hpgamma_source(jz);
            const type_real hlagrange = hxi_source(jx) * hgamma_source(jz);
            for (int ic = 0; ic < ncomponents; ++ic) {
              const type_real expected = source_tensor(ic, 0) * dsrc +
                                         source_tensor(ic, 1) * dsrc +
                                         hlagrange * body_couple(ic);
              EXPECT_NEAR(source_array(ic, jz, jx), expected, 1e-5)
                  << "Component " << ic << " at GLL point (" << jx << "," << jz
                  << ")";
            }
          }
        }
      }
    }
  }
}

// A zero body-couple vector is non-empty (does NOT opt out of the monopole
// path) but contributes nothing: the dipole-only result must be preserved
// exactly. This distinguishes the zero-couple case from the empty case
// (issue #2111).
TEST(ASSEMBLY_NO_LOAD, accumulate_monopole_zero_body_couple_adds_nothing) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Non-empty, all-zero body couple: must not opt out, must add nothing.
  BodyCoupleTestSource source({ { 1.0, 0.5 }, { 0.5, 2.0 }, { 0.0, 0.0 } },
                              { 0.0, 0.0, 0.0 });
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 3u);

  const auto source_tensor = source.get_source_tensor();
  const int ncomponents = source_tensor.extent(0);

  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      dipole_only("dipole_only", ncomponents, ngll, ngll);
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      with_monopole("with_monopole", ncomponents, ngll, ngll);

  std::vector<type_real> test_points = { xi_gamma_points(0), 0.0,
                                         xi_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real gamma_source : test_points) {
      SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                   ", gamma=" + std::to_string(gamma_source) + ")");
      source.set_local_coordinates(specfem::point::local_coordinates<
                                   specfem::element::dimension_tag::dim2>(
          0, xi_source, gamma_source));

      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            dipole_only(ic, jz, jx) = 0.0;
            with_monopole(ic, jz, jx) = 0.0;
          }
        }
      }

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
          for (int jx = 0; jx < ngll; ++jx) {
            EXPECT_NEAR(with_monopole(ic, jz, jx), dipole_only(ic, jz, jx),
                        1e-12)
                << "Zero body couple must not change the dipole result at "
                << "component " << ic << " point (" << jx << "," << jz << ")";
          }
        }
      }
    }
  }
}

// Spike test (level 1): a pure body couple (0, 0, c) applied through the
// monopole path must reproduce, exactly, a cosserat_force with f=0, fc=c
// applied through the vector (monopole) path. cosserat_force is the trusted
// reference for the monopole term (issue #2111).
TEST(ASSEMBLY_NO_LOAD, moment_tensor_monopole_matches_cosserat_force) {

  const int ngll = 5;
  const int ncomponents = 3;
  const type_real c = 1.75;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Zero tensor, body couple (0, 0, c): contributes only through the monopole.
  BodyCoupleTestSource couple_source(
      { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } }, { 0.0, 0.0, c });

  // Trusted reference: cosserat force with no elastic part (f = 0, fc = c).
  specfem::sources::cosserat_force<specfem::element::dimension_tag::dim2>
      force_source(0.0, 0.0, 0.0, c, 0.0,
                   std::make_unique<specfem::source_time_functions::Ricker>(
                       10, 0.01, 1.0, 0.0, 1.0, false),
                   specfem::simulation::field_type::forward);
  force_source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);

  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      couple_array("couple_array", ncomponents, ngll, ngll);
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      force_array("force_array", ncomponents, ngll, ngll);

  std::vector<type_real> test_points = { xi_gamma_points(0), -0.5, 0.0, 0.5,
                                         xi_gamma_points(ngll - 1) };

  for (type_real xi_source : test_points) {
    for (type_real gamma_source : test_points) {
      SCOPED_TRACE("source at (xi=" + std::to_string(xi_source) +
                   ", gamma=" + std::to_string(gamma_source) + ")");
      const auto local_coords = specfem::point::local_coordinates<
          specfem::element::dimension_tag::dim2>(0, xi_source, gamma_source);
      couple_source.set_local_coordinates(local_coords);
      force_source.set_local_coordinates(local_coords);

      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            couple_array(ic, jz, jx) = 0.0;
            force_array(ic, jz, jx) = 0.0;
          }
        }
      }

      specfem::assembly::compute_source_array_impl::
          compute_source_array_from_tensor_and_element_jacobian(
              couple_source, element_jacobian, quadrature, couple_array);
      specfem::assembly::compute_source_array_impl::
          accumulate_vector_contribution(couple_source.get_local_coordinates(),
                                         couple_source.get_body_couple_vector(),
                                         couple_array);
      specfem::assembly::compute_source_array_impl::from_vector(force_source,
                                                                force_array);

      for (int ic = 0; ic < ncomponents; ++ic) {
        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            EXPECT_NEAR(couple_array(ic, jz, jx), force_array(ic, jz, jx), 1e-5)
                << "Monopole body couple must match cosserat_force at "
                << "component " << ic << " point (" << jx << "," << jz << ")";
          }
        }
      }
    }
  }
}

// Spike test (level 2): an asymmetric moment tensor drives both the
// displacement rows (dipole M*grad L) and the rotation row (monopole
// L*(Mxz-Mzx)). Verifies the full combined result and pins the couple sign
// (issue #2112).
TEST(ASSEMBLY_NO_LOAD, asymmetric_moment_tensor_dipole_plus_monopole) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // (Mxx, Mzz, Mxz, Mzx) variants: a general asymmetric case and a one-sided
  // case to pin the couple sign.
  struct Params {
    type_real Mxx, Mzz, Mxz, Mzx;
  };
  std::vector<Params> cases = { { 0.8, 1.2, 0.3, -0.7 },
                                { 0.0, 0.0, 1.5, 0.0 } };

  for (const auto &p : cases) {
    SCOPED_TRACE("Mxz=" + std::to_string(p.Mxz) +
                 " Mzx=" + std::to_string(p.Mzx));
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
        source(0.0, 0.0, p.Mxx, p.Mzz, p.Mxz, p.Mzx,
               std::make_unique<specfem::source_time_functions::Ricker>(
                   10, 0.01, 1.0, 0.0, 1.0, false),
               specfem::simulation::field_type::forward);
    source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);

    const auto source_tensor = source.get_source_tensor();
    const int ncomponents = source_tensor.extent(0);
    ASSERT_EQ(ncomponents, 3);
    const type_real couple = p.Mxz - p.Mzx;

    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array("source_array", ncomponents, ngll, ngll);

    std::vector<type_real> test_points = { xi_gamma_points(0), 0.0, 0.5,
                                           xi_gamma_points(ngll - 1) };

    for (type_real xi_source : test_points) {
      for (type_real gamma_source : test_points) {
        source.set_local_coordinates(specfem::point::local_coordinates<
                                     specfem::element::dimension_tag::dim2>(
            0, xi_source, gamma_source));

        for (int ic = 0; ic < ncomponents; ++ic) {
          for (int jz = 0; jz < ngll; ++jz) {
            for (int jx = 0; jx < ngll; ++jx) {
              source_array(ic, jz, jx) = 0.0;
            }
          }
        }

        specfem::assembly::compute_source_array_impl::
            compute_source_array_from_tensor_and_element_jacobian(
                source, element_jacobian, quadrature, source_array);
        specfem::assembly::compute_source_array_impl::
            accumulate_vector_contribution(source.get_local_coordinates(),
                                           source.get_body_couple_vector(),
                                           source_array);

        auto [hxi_source, hpxi_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                xi_source, ngll, xi_gamma_points);
        auto [hgamma_source, hpgamma_source] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                gamma_source, ngll, xi_gamma_points);

        for (int jz = 0; jz < ngll; ++jz) {
          for (int jx = 0; jx < ngll; ++jx) {
            const type_real dsrc = hpxi_source(jx) * hgamma_source(jz) +
                                   hxi_source(jx) * hpgamma_source(jz);
            const type_real hlagrange = hxi_source(jx) * hgamma_source(jz);

            // Rows 0 and 1: displacement dipole only.
            EXPECT_NEAR(source_array(0, jz, jx),
                        source_tensor(0, 0) * dsrc + source_tensor(0, 1) * dsrc,
                        1e-5);
            EXPECT_NEAR(source_array(1, jz, jx),
                        source_tensor(1, 0) * dsrc + source_tensor(1, 1) * dsrc,
                        1e-5);
            // Row 2: rotation driven by the monopole body couple only (the
            // tensor's rotation row is zero).
            EXPECT_NEAR(source_array(2, jz, jx), hlagrange * couple, 1e-5);
          }
        }
      }
    }
  }
}

// Regression: a symmetric moment tensor (Mxz == Mzx) produces zero rotational
// coupling through the full tensor path (issue #2112).
TEST(ASSEMBLY_NO_LOAD, symmetric_moment_tensor_has_zero_rotational_coupling) {

  const int ngll = 5;

  specfem::quadrature::gll::gll gll_quad(0.0, 0.0, ngll);
  specfem::quadrature::quadratures quadratures(gll_quad);
  specfem::assembly::mesh_impl::quadrature<
      specfem::element::dimension_tag::dim2>
      quadrature(quadratures);
  auto xi_gamma_points = quadrature.h_xi;

  using PointJacobianMatrix =
      specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                      false, false>;
  Kokkos::View<PointJacobianMatrix **, Kokkos::LayoutRight, Kokkos::HostSpace>
      element_jacobian("element_jacobian", ngll, ngll);
  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      element_jacobian(iz, ix) = PointJacobianMatrix(1.0, 1.0, 1.0, 1.0);
    }
  }

  // Symmetric tensor (5-arg ctor sets Mzx = Mxz).
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2> source(
      0.0, 0.0, 1.0, 2.0, 0.5,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      specfem::simulation::field_type::forward);
  source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
  EXPECT_NEAR(source.get_body_couple_vector()(2), 0.0, 1e-12);

  const int ncomponents = source.get_source_tensor().extent(0);
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_array("source_array", ncomponents, ngll, ngll);

  source.set_local_coordinates(
      specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>(
          0, 0.3, -0.4));
  for (int ic = 0; ic < ncomponents; ++ic) {
    for (int jz = 0; jz < ngll; ++jz) {
      for (int jx = 0; jx < ngll; ++jx) {
        source_array(ic, jz, jx) = 0.0;
      }
    }
  }

  specfem::assembly::compute_source_array_impl::
      compute_source_array_from_tensor_and_element_jacobian(
          source, element_jacobian, quadrature, source_array);
  specfem::assembly::compute_source_array_impl::accumulate_vector_contribution(
      source.get_local_coordinates(), source.get_body_couple_vector(),
      source_array);

  // Rotation row must be identically zero.
  for (int jz = 0; jz < ngll; ++jz) {
    for (int jx = 0; jx < ngll; ++jx) {
      EXPECT_NEAR(source_array(2, jz, jx), 0.0, 1e-12);
    }
  }
}
