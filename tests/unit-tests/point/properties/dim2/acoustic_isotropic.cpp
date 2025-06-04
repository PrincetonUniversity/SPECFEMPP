#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 2D Acoustic Tests
// ============================================================================
TEST_F(PointPropertiesTest, AcousticIsotropic2D) {

  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr bool using_simd =
      specfem::datatype::simd<type_real, true>::using_simd;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Declare variables for properties
  simd_type rho;
  simd_type rho_inv;
  simd_type vp;
  simd_type kappa;
  simd_type kappa_inv;
  simd_type rho_vpinv;

  if (using_simd) {

    for (int i = 0; i < simd_size; ++i) {
      rho[i] = 1000.0 + i * 20.0; // kg/m³
      vp[i] = 1500.0 + i * 50.0;  // m/s
      kappa[i] = static_cast<type_real>(rho[i]) *
                 static_cast<type_real>(vp[i]) * static_cast<type_real>(vp[i]);
      rho_inv[i] = 1.0 / static_cast<type_real>(rho[i]);
      kappa_inv[i] = 1.0 / static_cast<type_real>(kappa[i]);
      rho_vpinv[i] = 1.0 / (static_cast<type_real>(rho[i]) *
                            static_cast<type_real>(vp[i]));
    }

  } else {

    // Water-like material
    constexpr type_real rho = 1000.0;          // kg/m³
    constexpr type_real vp = 1500.0;           // m/s
    constexpr type_real kappa = rho * vp * vp; // bulk modulus
    constexpr type_real rho_inv = 1.0 / rho;
    constexpr type_real kappa_inv = 1.0 / kappa;
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, using_simd>
      props(rho_inv, kappa);

  // Test accessors
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_inverse() - rho_inv) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.kappa() - kappa) < tol));

  // Test computed properties
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.kappa_inverse() - kappa_inv) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_vpinverse() - rho_vpinv) < tol));
}
