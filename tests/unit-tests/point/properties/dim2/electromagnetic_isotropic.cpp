#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 2D Electromagnetic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElectromagneticIsotropic2D) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr bool using_simd =
      specfem::datatype::simd<type_real, true>::using_simd;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Declare variables for properties
  simd_type mu0_inv;
  simd_type eps11;
  simd_type eps33;
  simd_type sig11;
  simd_type sig33;

  if (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      mu0_inv[i] = 7.957747e5 + i * 1.0e4; // 1/μ₀ (1/H/m)
      eps11[i] = 8.85e-12 + i * 1.0e-13;   // permittivity in xx (F/m)
      eps33[i] = 8.85e-12 + i * 1.0e-13;   // permittivity in zz (F/m)
      sig11[i] = 1.0e-2 + i * 1.0e-3;      // conductivity in xx (S/m)
      sig33[i] = 1.0e-2 + i * 1.0e-3;      // conductivity in zz (S/m)
    }
  } else {
    // Electromagnetic medium properties for scalar case
    mu0_inv = 7.957747e5; // 1/μ₀ (1/H/m)
    eps11 = 8.85e-12;     // permittivity in xx (F/m)
    eps33 = 8.85e-12;     // permittivity in zz (F/m)
    sig11 = 1.0e-2;       // conductivity in xx (S/m)
    sig33 = 1.0e-2;       // conductivity in zz (S/m)
  }

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::electromagnetic,
                             specfem::element::property_tag::isotropic,
                             using_simd>
      props(mu0_inv, eps11, eps33, sig11, sig33);

  // Test accessors with tolerance-based comparisons
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.mu0_inv() - mu0_inv) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.eps11() - eps11) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.eps33() - eps33) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.sig11() - sig11) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.sig33() - sig33) < tol));
}
