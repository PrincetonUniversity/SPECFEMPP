#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Acoustic Tests
// ============================================================================
TEST_F(PointKernelsTest, AcousticIsotropic2D) {
  // Create test kernel values
  constexpr type_real rho = 2.5;
  constexpr type_real kappa = 3.0;

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, false>
      kernels(rho, kappa);

  // Test accessors
  EXPECT_REAL_EQ(kernels.rho(), rho);
  EXPECT_REAL_EQ(kernels.kappa(), kappa);

  // Test computed values based on constructor
  EXPECT_REAL_EQ(kernels.rhop(), rho * kappa);
  EXPECT_REAL_EQ(kernels.alpha(), static_cast<type_real>(2.0) * kappa);
}

// Test SIMD version of the kernels
TEST_F(PointKernelsTest, AcousticIsotropic2D_SIMD) {
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
  simd_type kappa_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> kappa_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2.5 + i * 0.1;
    kappa_values[i] = 3.0 + i * 0.1;

    // Load into SIMD vectors using operator[]
    rho_simd[i] = rho_values[i];
    kappa_simd[i] = kappa_values[i];
  }

  // Create the kernels object with SIMD data
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, true>
      kernels(rho_simd, kappa_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real rho = kernels.rho()[lane];
    type_real kappa = kernels.kappa()[lane];
    type_real rhop = kernels.rhop()[lane];
    type_real alpha = kernels.alpha()[lane];

    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(kappa, kappa_values[lane]);
    EXPECT_REAL_EQ(rhop, rho_values[lane] * kappa_values[lane]);
    EXPECT_REAL_EQ(alpha, static_cast<type_real>(2.0) * kappa_values[lane]);
  }
}
