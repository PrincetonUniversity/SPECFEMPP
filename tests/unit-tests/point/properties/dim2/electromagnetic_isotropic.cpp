#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Electromagnetic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElectromagneticIsotropic2D) {
  // Electromagnetic medium properties
  constexpr type_real mu0_inv = 7.957747e5; // 1/μ₀ (1/H/m)
  constexpr type_real eps11 = 8.85e-12;     // permittivity in xx (F/m)
  constexpr type_real eps33 = 8.85e-12;     // permittivity in zz (F/m)
  constexpr type_real sig11 = 1.0e-2;       // conductivity in xx (S/m)
  constexpr type_real sig33 = 1.0e-2;       // conductivity in zz (S/m)

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::electromagnetic,
                             specfem::element::property_tag::isotropic, false>
      props(mu0_inv, eps11, eps33, sig11, sig33);

  // Test accessors
  EXPECT_REAL_EQ(props.mu0_inv(), mu0_inv);
  EXPECT_REAL_EQ(props.eps11(), eps11);
  EXPECT_REAL_EQ(props.eps33(), eps33);
  EXPECT_REAL_EQ(props.sig11(), sig11);
  EXPECT_REAL_EQ(props.sig33(), sig33);
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, ElectromagneticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type mu0_inv_simd;
  simd_type eps11_simd;
  simd_type eps33_simd;
  simd_type sig11_simd;
  simd_type sig33_simd;

  // Setup test data
  std::vector<type_real> mu0_inv_values(simd_size);
  std::vector<type_real> eps11_values(simd_size);
  std::vector<type_real> eps33_values(simd_size);
  std::vector<type_real> sig11_values(simd_size);
  std::vector<type_real> sig33_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    mu0_inv_values[i] =
        static_cast<type_real>(7.957747e5 + i * 1.0e4);               // (1/H/m)
    eps11_values[i] = static_cast<type_real>(8.85e-12 + i * 1.0e-13); // (F/m)
    eps33_values[i] = static_cast<type_real>(8.85e-12 + i * 1.0e-13); // (F/m)
    sig11_values[i] = static_cast<type_real>(1.0e-2 + i * 1.0e-3);    // (S/m)
    sig33_values[i] = static_cast<type_real>(1.0e-2 + i * 1.0e-3);    // (S/m)

    // Load into SIMD vectors using operator[]
    mu0_inv_simd[i] = mu0_inv_values[i];
    eps11_simd[i] = eps11_values[i];
    eps33_simd[i] = eps33_values[i];
    sig11_simd[i] = sig11_values[i];
    sig33_simd[i] = sig33_values[i];
  }
  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::electromagnetic,
                             specfem::element::property_tag::isotropic, true>
      props(mu0_inv_simd, eps11_simd, eps33_simd, sig11_simd, sig33_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real mu0_inv = props.mu0_inv()[lane];
    type_real eps11 = props.eps11()[lane];
    type_real eps33 = props.eps33()[lane];
    type_real sig11 = props.sig11()[lane];
    type_real sig33 = props.sig33()[lane];

    EXPECT_REAL_EQ(mu0_inv, mu0_inv_values[lane]);
    EXPECT_REAL_EQ(eps11, eps11_values[lane]);
    EXPECT_REAL_EQ(eps33, eps33_values[lane]);
    EXPECT_REAL_EQ(sig11, sig11_values[lane]);
    EXPECT_REAL_EQ(sig33, sig33_values[lane]);
  }
}
