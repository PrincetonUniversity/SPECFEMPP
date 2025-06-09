#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 3D Elastic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticIsotropic3D) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr bool using_simd =
      specfem::datatype::simd<type_real, true>::using_simd;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Declare variables for properties
  simd_type kappa;
  simd_type mu;
  simd_type rho;
  simd_type lambdaplus2mu_val;
  simd_type lambda_val;
  simd_type rho_vp_val;
  simd_type rho_vs_val;

  if (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = 2700.0 + i * 50.0;              // kg/m³
      const type_real vp = 6000.0 + i * 100.0; // m/s
      const type_real vs = 3500.0 + i * 50.0;  // m/s

      mu[i] = rho[i] * vs * vs;
      kappa[i] = rho[i] * (vp * vp - (4.0 / 3.0) * vs * vs);
      lambdaplus2mu_val[i] = kappa[i] + (4.0 / 3.0) * mu[i];
      lambda_val[i] = lambdaplus2mu_val[i] - 2.0 * mu[i];
      rho_vp_val[i] = rho[i] * vp;
      rho_vs_val[i] = rho[i] * vs;
    }
  } else {
    // Granite-like material for scalar test
    constexpr type_real rho_val = 2700.0;           // kg/m³
    constexpr type_real vp = 6000.0;                // m/s
    constexpr type_real vs = 3500.0;                // m/s
    constexpr type_real mu_val = rho_val * vs * vs; // shear modulus

    // For 3D, we use kappa (bulk modulus) instead of lambda + 2*mu
    constexpr type_real kappa_val =
        rho_val * (vp * vp - (4.0 / 3.0) * vs * vs); // bulk modulus
    constexpr type_real lambdaplus2mu_scalar = kappa_val + (4.0 / 3.0) * mu_val;
    constexpr type_real lambda_scalar = lambdaplus2mu_scalar - 2.0 * mu_val;

    // Assign to our variables
    kappa = kappa_val;
    mu = mu_val;
    rho = rho_val;
    lambdaplus2mu_val = lambdaplus2mu_scalar;
    lambda_val = lambda_scalar;
    rho_vp_val = rho_val * vp;
    rho_vs_val = rho_val * vs;
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>
      props(kappa, mu, rho);

  // Test accessors with tolerance-based comparisons
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.kappa() - kappa) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.mu() - mu) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.rho() - rho) < tol));

  // Test computed properties
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambdaplus2mu() - lambdaplus2mu_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambda() - lambda_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_vp() - rho_vp_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_vs() - rho_vs_val) < tol));
}
