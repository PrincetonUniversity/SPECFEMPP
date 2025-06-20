#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Anisotropic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, ElasticAnisotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Declare variables for properties
  simd_type c11;
  simd_type c13;
  simd_type c15;
  simd_type c33;
  simd_type c35;
  simd_type c55;
  simd_type c12;
  simd_type c23;
  simd_type c25;
  simd_type rho;
  simd_type rho_vp_val;
  simd_type rho_vs_val;

  if constexpr (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = 2500.0 + i * 50.0;  // kg/m³
      c11[i] = 75.0e9 + i * 1.0e9; // Pa
      c13[i] = 15.0e9 + i * 0.5e9; // Pa
      c15[i] = i * 0.0;            // Pa (zero for simplicity)
      c33[i] = 55.0e9 + i * 1.0e9; // Pa
      c35[i] = i * 0.0;            // Pa (zero for simplicity)
      c55[i] = 20.0e9 + i * 0.5e9; // Pa
      c12[i] = 15.0e9 + i * 0.5e9; // Pa
      c23[i] = 10.0e9 + i * 0.5e9; // Pa
      c25[i] = i * 0.0;            // Pa (zero for simplicity)

      // Computed values for verification
      rho_vp_val[i] = std::sqrt(static_cast<type_real>(rho[i]) *
                                static_cast<type_real>(c33[i]));
      rho_vs_val[i] = std::sqrt(static_cast<type_real>(rho[i]) *
                                static_cast<type_real>(c55[i]));
    }
  } else {
    // Anisotropic material values (e.g., shale-like)
    constexpr type_real rho_val = 2500.0; // kg/m³
    constexpr type_real c11_val = 75.0e9; // Pa
    constexpr type_real c13_val = 15.0e9; // Pa
    constexpr type_real c15_val = 0.0;    // Pa (zero for simplicity)
    constexpr type_real c33_val = 55.0e9; // Pa
    constexpr type_real c35_val = 0.0;    // Pa (zero for simplicity)
    constexpr type_real c55_val = 20.0e9; // Pa
    constexpr type_real c12_val = 15.0e9; // Pa
    constexpr type_real c23_val = 10.0e9; // Pa
    constexpr type_real c25_val = 0.0;    // Pa (zero for simplicity)

    // Assign to our variables
    rho = rho_val;
    c11 = c11_val;
    c13 = c13_val;
    c15 = c15_val;
    c33 = c33_val;
    c35 = c35_val;
    c55 = c55_val;
    c12 = c12_val;
    c23 = c23_val;
    c25 = c25_val;

    // Computed values for verification
    rho_vp_val = std::sqrt(rho_val * c33_val);
    rho_vs_val = std::sqrt(rho_val * c55_val);
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, using_simd>
      props(c11, c13, c15, c33, c35, c55, c12, c23, c25, rho);

  EXPECT_TRUE(specfem::utilities::is_close(props.c11(), c11))
      << ExpectedGot(c11, props.c11());
  EXPECT_TRUE(specfem::utilities::is_close(props.c13(), c13))
      << ExpectedGot(c13, props.c13());
  EXPECT_TRUE(specfem::utilities::is_close(props.c15(), c15))
      << ExpectedGot(c15, props.c15());
  EXPECT_TRUE(specfem::utilities::is_close(props.c33(), c33))
      << ExpectedGot(c33, props.c33());
  EXPECT_TRUE(specfem::utilities::is_close(props.c35(), c35))
      << ExpectedGot(c35, props.c35());
  EXPECT_TRUE(specfem::utilities::is_close(props.c55(), c55))
      << ExpectedGot(c55, props.c55());
  EXPECT_TRUE(specfem::utilities::is_close(props.c12(), c12))
      << ExpectedGot(c12, props.c12());
  EXPECT_TRUE(specfem::utilities::is_close(props.c23(), c23))
      << ExpectedGot(c23, props.c23());
  EXPECT_TRUE(specfem::utilities::is_close(props.c25(), c25))
      << ExpectedGot(c25, props.c25());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho(), rho))
      << ExpectedGot(rho, props.rho());

  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vp(), rho_vp_val))
      << ExpectedGot(rho_vp_val, props.rho_vp());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_vs(), rho_vs_val))
      << ExpectedGot(rho_vs_val, props.rho_vs());
}
