#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

#ifdef ENABLE_SIMD
bool USE_SIMD = true; ///< Flag to enable SIMD operations

#else
bool USE_SIMD = false; ///< Flag to disable SIMD operations
#endif

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticIsotropic2D) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr bool using_simd =
      specfem::datatype::simd<type_real, true>::using_simd;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Declare variables for properties
  simd_type lambdaplus2mu;
  simd_type mu;
  simd_type rho;
  simd_type lambda;
  simd_type rho_vp;
  simd_type rho_vs;

  if (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = 2700.0 + i * 50.0;              // kg/m³
      const type_real vp = 6000.0 + i * 100.0; // m/s
      const type_real vs = 3500.0 + i * 50.0;  // m/s

      mu[i] = rho[i] * vs * vs;
      lambda[i] = static_cast<type_real>(rho[i]) * vp * vp - 2.0 * mu[i];
      lambdaplus2mu[i] = static_cast<type_real>(lambda[i]) +
                         2.0 * static_cast<type_real>(mu[i]);
      rho_vp[i] = static_cast<type_real>(rho[i]) * vp;
      rho_vs[i] = static_cast<type_real>(rho[i]) * vs;
    }
  } else {
    // Granite-like material for scalar test
    constexpr type_real rho_val = 2700.0;           // kg/m³
    constexpr type_real vp = 6000.0;                // m/s
    constexpr type_real vs = 3500.0;                // m/s
    constexpr type_real mu_val = rho_val * vs * vs; // shear modulus
    constexpr type_real lambda_val =
        rho_val * vp * vp - 2 * mu_val; // first Lamé parameter
    constexpr type_real lambdaplus2mu_val = lambda_val + 2 * mu_val;

    // Assign to our variables
    rho = rho_val;
    mu = mu_val;
    lambdaplus2mu = lambdaplus2mu_val;
    lambda = lambda_val;
    rho_vp = rho_val * vp;
    rho_vs = rho_val * vs;
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>
      props(lambdaplus2mu, mu, rho);

  // Test accessors with tolerance-based comparisons
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambdaplus2mu() - lambdaplus2mu) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.mu() - mu) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.rho() - rho) < tol));

  // Test computed properties
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.lambda() - lambda) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_vp() - rho_vp) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_vs() - rho_vs) < tol));
}
