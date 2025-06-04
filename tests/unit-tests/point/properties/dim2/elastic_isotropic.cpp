#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticIsotropic2D) {
  // Granite-like material
  constexpr type_real rho = 2700.0;                    // kg/m³
  constexpr type_real vp = 6000.0;                     // m/s
  constexpr type_real vs = 3500.0;                     // m/s
  constexpr type_real mu = rho * vs * vs;              // shear modulus
  constexpr type_real lambda = rho * vp * vp - 2 * mu; // first Lamé parameter
  constexpr type_real lambdaplus2mu = lambda + 2 * mu;

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, false>
      props(lambdaplus2mu, mu, rho);

  // Test accessors
  EXPECT_REAL_EQ(props.lambdaplus2mu(), lambdaplus2mu);
  EXPECT_REAL_EQ(props.mu(), mu);
  EXPECT_REAL_EQ(props.rho(), rho);

  // Test computed properties
  EXPECT_REAL_EQ(props.lambda(), lambda);
  EXPECT_REAL_EQ(props.rho_vp(), rho * vp);
  EXPECT_REAL_EQ(props.rho_vs(), rho * vs);
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, ElasticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type lambdaplus2mu_simd;
  simd_type mu_simd;
  simd_type rho_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> vp_values(simd_size);
  std::vector<type_real> vs_values(simd_size);
  std::vector<type_real> mu_values(simd_size);
  std::vector<type_real> lambda_values(simd_size);
  std::vector<type_real> lambdaplus2mu_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2700.0 + i * 50.0; // kg/m³
    vp_values[i] = 6000.0 + i * 100.0; // m/s
    vs_values[i] = 3500.0 + i * 50.0;  // m/s
    mu_values[i] = rho_values[i] * vs_values[i] * vs_values[i];
    lambda_values[i] =
        rho_values[i] * vp_values[i] * vp_values[i] - 2.0 * mu_values[i];
    lambdaplus2mu_values[i] = lambda_values[i] + 2.0 * mu_values[i];

    // Load into SIMD vectors
    lambdaplus2mu_simd[i] = lambdaplus2mu_values[i];
    mu_simd[i] = mu_values[i];
    rho_simd[i] = rho_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, true>
      props(lambdaplus2mu_simd, mu_simd, rho_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real lambdaplus2mu = props.lambdaplus2mu()[lane];
    type_real mu = props.mu()[lane];
    type_real rho = props.rho()[lane];
    type_real lambda = props.lambda()[lane];
    type_real rho_vp = props.rho_vp()[lane];
    type_real rho_vs = props.rho_vs()[lane];

    EXPECT_REAL_EQ(lambdaplus2mu, lambdaplus2mu_values[lane]);
    EXPECT_REAL_EQ(mu, mu_values[lane]);
    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(lambda, lambda_values[lane]);
    EXPECT_REAL_EQ(rho_vp, rho_values[lane] * vp_values[lane]);
    EXPECT_REAL_EQ(rho_vs, rho_values[lane] * vs_values[lane]);
  }
}
