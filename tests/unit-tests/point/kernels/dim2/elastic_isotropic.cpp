#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TEST_F(PointKernelsTest, ElasticIsotropic2D) {
  // Create test kernel values
  constexpr type_real rho = 2.5;
  constexpr type_real mu = 3.0;
  constexpr type_real kappa = 4.0;
  constexpr type_real rhop = 5.0;
  constexpr type_real alpha = 6.0;
  constexpr type_real beta = 7.0;

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, false>
      kernels(rho, mu, kappa, rhop, alpha, beta);

  // Test accessors
  EXPECT_REAL_EQ(kernels.rho(), rho);
  EXPECT_REAL_EQ(kernels.mu(), mu);
  EXPECT_REAL_EQ(kernels.kappa(), kappa);
  EXPECT_REAL_EQ(kernels.rhop(), rhop);
  EXPECT_REAL_EQ(kernels.alpha(), alpha);
  EXPECT_REAL_EQ(kernels.beta(), beta);
}

// Test SIMD version of the kernels
TEST_F(PointKernelsTest, ElasticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type rho_simd;
  simd_type mu_simd;
  simd_type kappa_simd;
  simd_type rhop_simd;
  simd_type alpha_simd;
  simd_type beta_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> mu_values(simd_size);
  std::vector<type_real> kappa_values(simd_size);
  std::vector<type_real> rhop_values(simd_size);
  std::vector<type_real> alpha_values(simd_size);
  std::vector<type_real> beta_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2.5 + i * 0.1;
    mu_values[i] = 3.0 + i * 0.1;
    kappa_values[i] = 4.0 + i * 0.1;
    rhop_values[i] = 5.0 + i * 0.1;
    alpha_values[i] = 6.0 + i * 0.1;
    beta_values[i] = 7.0 + i * 0.1;

    // Load into SIMD vectors using operator[]
    rho_simd[i] = rho_values[i];
    mu_simd[i] = mu_values[i];
    kappa_simd[i] = kappa_values[i];
    rhop_simd[i] = rhop_values[i];
    alpha_simd[i] = alpha_values[i];
    beta_simd[i] = beta_values[i];
  }

  // Create the kernels object with SIMD data
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, true>
      kernels(rho_simd, mu_simd, kappa_simd, rhop_simd, alpha_simd, beta_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real rho = kernels.rho()[lane];
    type_real mu = kernels.mu()[lane];
    type_real kappa = kernels.kappa()[lane];
    type_real rhop = kernels.rhop()[lane];
    type_real alpha = kernels.alpha()[lane];
    type_real beta = kernels.beta()[lane];

    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(mu, mu_values[lane]);
    EXPECT_REAL_EQ(kappa, kappa_values[lane]);
    EXPECT_REAL_EQ(rhop, rhop_values[lane]);
    EXPECT_REAL_EQ(alpha, alpha_values[lane]);
    EXPECT_REAL_EQ(beta, beta_values[lane]);
  }
}
