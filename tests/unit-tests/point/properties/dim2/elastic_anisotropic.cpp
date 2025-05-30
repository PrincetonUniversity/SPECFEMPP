#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Anisotropic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticAnisotropic2D) {
  // Anisotropic material values (e.g., shale-like)
  constexpr type_real rho = 2500.0; // kg/m³
  constexpr type_real c11 = 75.0e9; // Pa
  constexpr type_real c13 = 15.0e9; // Pa
  constexpr type_real c15 = 0.0;    // Pa (zero for simplicity)
  constexpr type_real c33 = 55.0e9; // Pa
  constexpr type_real c35 = 0.0;    // Pa (zero for simplicity)
  constexpr type_real c55 = 20.0e9; // Pa
  constexpr type_real c12 = 15.0e9; // Pa
  constexpr type_real c23 = 10.0e9; // Pa
  constexpr type_real c25 = 0.0;    // Pa (zero for simplicity)

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::anisotropic, false>
      props(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho);

  // Test accessors
  EXPECT_REAL_EQ(props.c11(), c11);
  EXPECT_REAL_EQ(props.c13(), c13);
  EXPECT_REAL_EQ(props.c15(), c15);
  EXPECT_REAL_EQ(props.c33(), c33);
  EXPECT_REAL_EQ(props.c35(), c35);
  EXPECT_REAL_EQ(props.c55(), c55);
  EXPECT_REAL_EQ(props.c12(), c12);
  EXPECT_REAL_EQ(props.c23(), c23);
  EXPECT_REAL_EQ(props.c25(), c25);
  EXPECT_REAL_EQ(props.rho(), rho);

  // Test computed properties
  EXPECT_REAL_EQ(props.rho_vp(), std::sqrt(rho * c33));
  EXPECT_REAL_EQ(props.rho_vs(), std::sqrt(rho * c55));
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, ElasticAnisotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type c11_simd;
  simd_type c13_simd;
  simd_type c15_simd;
  simd_type c33_simd;
  simd_type c35_simd;
  simd_type c55_simd;
  simd_type c12_simd;
  simd_type c23_simd;
  simd_type c25_simd;
  simd_type rho_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> c11_values(simd_size);
  std::vector<type_real> c13_values(simd_size);
  std::vector<type_real> c15_values(simd_size);
  std::vector<type_real> c33_values(simd_size);
  std::vector<type_real> c35_values(simd_size);
  std::vector<type_real> c55_values(simd_size);
  std::vector<type_real> c12_values(simd_size);
  std::vector<type_real> c23_values(simd_size);
  std::vector<type_real> c25_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2500.0 + i * 50.0;  // kg/m³
    c11_values[i] = 75.0e9 + i * 1.0e9; // Pa
    c13_values[i] = 15.0e9 + i * 0.5e9; // Pa
    c15_values[i] = i * 0.0;            // Pa (zero for simplicity)
    c33_values[i] = 55.0e9 + i * 1.0e9; // Pa
    c35_values[i] = i * 0.0;            // Pa (zero for simplicity)
    c55_values[i] = 20.0e9 + i * 0.5e9; // Pa
    c12_values[i] = 15.0e9 + i * 0.5e9; // Pa
    c23_values[i] = 10.0e9 + i * 0.5e9; // Pa
    c25_values[i] = i * 0.0;            // Pa (zero for simplicity)

    // Load into SIMD vectors
    c11_simd[i] = c11_values[i];
    c13_simd[i] = c13_values[i];
    c15_simd[i] = c15_values[i];
    c33_simd[i] = c33_values[i];
    c35_simd[i] = c35_values[i];
    c55_simd[i] = c55_values[i];
    c12_simd[i] = c12_values[i];
    c23_simd[i] = c23_values[i];
    c25_simd[i] = c25_values[i];
    rho_simd[i] = rho_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::anisotropic, true>
      props(c11_simd, c13_simd, c15_simd, c33_simd, c35_simd, c55_simd,
            c12_simd, c23_simd, c25_simd, rho_simd);
  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real c11 = props.c11()[lane];
    type_real c13 = props.c13()[lane];
    type_real c15 = props.c15()[lane];
    type_real c33 = props.c33()[lane];
    type_real c35 = props.c35()[lane];
    type_real c55 = props.c55()[lane];
    type_real c12 = props.c12()[lane];
    type_real c23 = props.c23()[lane];
    type_real c25 = props.c25()[lane];
    type_real rho = props.rho()[lane];

    EXPECT_REAL_EQ(c11, c11_values[lane]);
    EXPECT_REAL_EQ(c13, c13_values[lane]);
    EXPECT_REAL_EQ(c15, c15_values[lane]);
    EXPECT_REAL_EQ(c33, c33_values[lane]);
    EXPECT_REAL_EQ(c35, c35_values[lane]);
    EXPECT_REAL_EQ(c55, c55_values[lane]);
    EXPECT_REAL_EQ(c12, c12_values[lane]);
    EXPECT_REAL_EQ(c23, c23_values[lane]);
    EXPECT_REAL_EQ(c25, c25_values[lane]);
    EXPECT_REAL_EQ(rho, rho_values[lane]);

    // Test computed properties
    EXPECT_REAL_EQ(props.rho_vp()[lane], std::sqrt(rho * c33));
    EXPECT_REAL_EQ(props.rho_vs()[lane], std::sqrt(rho * c55));
  }
}
