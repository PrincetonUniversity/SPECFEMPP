#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Acoustic Tests
// ============================================================================
TEST_F(PointPropertiesTest, AcousticIsotropic2D) {
  // Water-like material
  constexpr type_real rho = 1000.0;          // kg/m³
  constexpr type_real vp = 1500.0;           // m/s
  constexpr type_real kappa = rho * vp * vp; // bulk modulus
  constexpr type_real rho_inv = 1.0 / rho;
  constexpr type_real kappa_inv = 1.0 / kappa;

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, false>
      props(rho_inv, kappa);

  // Test accessors
  EXPECT_REAL_EQ(props.rho_inverse(), rho_inv);
  EXPECT_REAL_EQ(props.kappa(), kappa);

  // Test computed properties
  EXPECT_REAL_EQ(props.kappa_inverse(), kappa_inv);
  EXPECT_REAL_EQ(props.rho_vpinverse(), 1.0 / (rho * vp));
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, AcousticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available (size == 1)
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type rho_inv_simd;
  simd_type kappa_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> vp_values(simd_size);
  std::vector<type_real> kappa_values(simd_size);
  std::vector<type_real> rho_inv_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 1000.0 + i * 20.0; // kg/m³
    vp_values[i] = 1500.0 + i * 50.0;  // m/s
    kappa_values[i] = rho_values[i] * vp_values[i] * vp_values[i];
    rho_inv_values[i] = 1.0 / rho_values[i];

    // Load into SIMD vectors using operator[]
    rho_inv_simd[i] = rho_inv_values[i];
    kappa_simd[i] = kappa_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::acoustic,
                             specfem::element::property_tag::isotropic, true>
      props(rho_inv_simd, kappa_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real rho_inv = props.rho_inverse()[lane];
    type_real kappa = props.kappa()[lane];
    type_real kappa_inv = props.kappa_inverse()[lane];
    type_real rho_vpinv = props.rho_vpinverse()[lane];

    EXPECT_REAL_EQ(rho_inv, rho_inv_values[lane]);
    EXPECT_REAL_EQ(kappa, kappa_values[lane]);
    EXPECT_REAL_EQ(kappa_inv, 1.0 / kappa_values[lane]);
    EXPECT_REAL_EQ(rho_vpinv, 1.0 / (rho_values[lane] * vp_values[lane]));
  }
}
