#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Anisotropic Tests
// ============================================================================
TEST_F(PointKernelsTest, ElasticAnisotropic2D) {
  // Create test kernel values
  constexpr type_real rho = 2.5;
  constexpr type_real c11 = 10.0;
  constexpr type_real c13 = 11.0;
  constexpr type_real c15 = 12.0;
  constexpr type_real c33 = 13.0;
  constexpr type_real c35 = 14.0;
  constexpr type_real c55 = 15.0;

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::anisotropic, false>
      kernels(rho, c11, c13, c15, c33, c35, c55);

  // Test accessors
  EXPECT_REAL_EQ(kernels.rho(), rho);
  EXPECT_REAL_EQ(kernels.c11(), c11);
  EXPECT_REAL_EQ(kernels.c13(), c13);
  EXPECT_REAL_EQ(kernels.c15(), c15);
  EXPECT_REAL_EQ(kernels.c33(), c33);
  EXPECT_REAL_EQ(kernels.c35(), c35);
  EXPECT_REAL_EQ(kernels.c55(), c55);
}

// Test SIMD version of the kernels
TEST_F(PointKernelsTest, ElasticAnisotropic2D_SIMD) {
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
  simd_type c11_simd;
  simd_type c13_simd;
  simd_type c15_simd;
  simd_type c33_simd;
  simd_type c35_simd;
  simd_type c55_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> c11_values(simd_size);
  std::vector<type_real> c13_values(simd_size);
  std::vector<type_real> c15_values(simd_size);
  std::vector<type_real> c33_values(simd_size);
  std::vector<type_real> c35_values(simd_size);
  std::vector<type_real> c55_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2.5 + i * 0.1;
    c11_values[i] = 10.0 + i * 0.1;
    c13_values[i] = 11.0 + i * 0.1;
    c15_values[i] = 12.0 + i * 0.1;
    c33_values[i] = 13.0 + i * 0.1;
    c35_values[i] = 14.0 + i * 0.1;
    c55_values[i] = 15.0 + i * 0.1;

    // Load into SIMD vectors using operator[]
    rho_simd[i] = rho_values[i];
    c11_simd[i] = c11_values[i];
    c13_simd[i] = c13_values[i];
    c15_simd[i] = c15_values[i];
    c33_simd[i] = c33_values[i];
    c35_simd[i] = c35_values[i];
    c55_simd[i] = c55_values[i];
  }

  // Create the kernels object with SIMD data
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::anisotropic, true>
      kernels(rho_simd, c11_simd, c13_simd, c15_simd, c33_simd, c35_simd,
              c55_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real rho = kernels.rho()[lane];
    type_real c11 = kernels.c11()[lane];
    type_real c13 = kernels.c13()[lane];
    type_real c15 = kernels.c15()[lane];
    type_real c33 = kernels.c33()[lane];
    type_real c35 = kernels.c35()[lane];
    type_real c55 = kernels.c55()[lane];

    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(c11, c11_values[lane]);
    EXPECT_REAL_EQ(c13, c13_values[lane]);
    EXPECT_REAL_EQ(c15, c15_values[lane]);
    EXPECT_REAL_EQ(c33, c33_values[lane]);
    EXPECT_REAL_EQ(c35, c35_values[lane]);
    EXPECT_REAL_EQ(c55, c55_values[lane]);
  }
}
