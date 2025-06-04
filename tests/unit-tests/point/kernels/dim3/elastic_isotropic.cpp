#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 3D Elastic Isotropic Tests
// ============================================================================
TEST_F(PointKernelsTest, ElasticIsotropic3D) {
  // Create test kernel values for 3D elastic isotropic medium
  constexpr type_real rho = 2.7;    // Density kernel
  constexpr type_real mu = 30.0;    // Shear modulus kernel
  constexpr type_real kappa = 40.0; // Bulk modulus kernel
  constexpr type_real rhop = 50.0;  // Density*P-velocity kernel
  constexpr type_real alpha = 60.0; // P-velocity kernel
  constexpr type_real beta = 70.0;  // S-velocity kernel

  // Create the kernels object for 3D elastic isotropic medium
  specfem::point::kernels<specfem::dimension::type::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, false>
      kernels(rho, mu, kappa, rhop, alpha, beta);

  // Test accessors (direct values)
  EXPECT_REAL_EQ(kernels.rho(), rho);
  EXPECT_REAL_EQ(kernels.mu(), mu);
  EXPECT_REAL_EQ(kernels.kappa(), kappa);
  EXPECT_REAL_EQ(kernels.rhop(), rhop);
  EXPECT_REAL_EQ(kernels.alpha(), alpha);
  EXPECT_REAL_EQ(kernels.beta(), beta);
}

// ============================================================================
// 3D Elastic Isotropic SIMD Tests
// ============================================================================
TEST_F(PointKernelsTest, ElasticIsotropic3D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects for 3D elastic isotropic kernels
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
    rho_values[i] = 2.7 + i * 0.1;    // Density kernel with variations
    mu_values[i] = 30.0 + i * 0.5;    // Shear modulus kernel with variations
    kappa_values[i] = 40.0 + i * 0.5; // Bulk modulus kernel with variations
    rhop_values[i] =
        50.0 + i * 1.0; // Density*P-velocity kernel with variations
    alpha_values[i] = 60.0 + i * 1.0; // P-velocity kernel with variations
    beta_values[i] = 70.0 + i * 1.0;  // S-velocity kernel with variations

    // Load into SIMD vectors using operator[]
    rho_simd[i] = rho_values[i];
    mu_simd[i] = mu_values[i];
    kappa_simd[i] = kappa_values[i];
    rhop_simd[i] = rhop_values[i];
    alpha_simd[i] = alpha_values[i];
    beta_simd[i] = beta_values[i];
  }

  // Create the kernels object with SIMD data
  specfem::point::kernels<specfem::dimension::type::dim3,
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
