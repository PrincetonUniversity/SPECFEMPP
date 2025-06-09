#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TEST_F(PointPropertiesTest, PoroelasticIsotropic2D) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr bool using_simd = simd_type::using_simd;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Declare variables for properties
  simd_type phi;
  simd_type rho_s;
  simd_type rho_f;
  simd_type tortuosity;
  simd_type mu_G;
  simd_type H_Biot;
  simd_type C_Biot;
  simd_type M_Biot;
  simd_type permxx;
  simd_type permxz;
  simd_type permzz;
  simd_type eta_f;

  // Variables for computed properties
  simd_type lambda_G_val;
  simd_type rho_bar_val;
  simd_type perm_det;
  simd_type inverse_permxx_val;
  simd_type inverse_permxz_val;
  simd_type inverse_permzz_val;
  simd_type vs_expected;

  if (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      phi[i] = 0.2 + i * 0.01;           // porosity
      rho_s[i] = 2650.0 + i * 10.0;      // solid density (kg/m³)
      rho_f[i] = 1000.0 + i * 5.0;       // fluid density (kg/m³)
      tortuosity[i] = 2.0 + i * 0.1;     // tortuosity
      mu_G[i] = 10.0e9 + i * 1.0e8;      // shear modulus (Pa)
      H_Biot[i] = 25.0e9 + i * 1.0e8;    // Biot's H modulus (Pa)
      C_Biot[i] = 10.0e9 + i * 1.0e7;    // Biot's C modulus (Pa)
      M_Biot[i] = 15.0e9 + i * 1.0e7;    // Biot's M modulus (Pa)
      permxx[i] = 1.0e-12 + i * 1.0e-14; // permeability in xx (m²)
      permxz[i] = 0.0;                   // permeability in xz (m²)
      permzz[i] = 1.0e-12 + i * 1.0e-14; // permeability in zz (m²)
      eta_f[i] = 1.0e-3 + i * 1.0e-5;    // fluid viscosity (Pa·s)

      // Compute expected properties for verification
      lambda_G_val[i] = static_cast<type_real>(H_Biot[i]) -
                        2.0 * static_cast<type_real>(mu_G[i]);
      rho_bar_val[i] =
          (1.0 - static_cast<type_real>(phi[i])) *
              static_cast<type_real>(rho_s[i]) +
          static_cast<type_real>(phi[i]) * static_cast<type_real>(rho_f[i]);
      perm_det[i] =
          static_cast<type_real>(permxx[i]) *
              static_cast<type_real>(permzz[i]) -
          static_cast<type_real>(permxz[i]) * static_cast<type_real>(permxz[i]);
      inverse_permxx_val[i] = static_cast<type_real>(permzz[i]) /
                              static_cast<type_real>(perm_det[i]);
      inverse_permxz_val[i] = -static_cast<type_real>(permxz[i]) /
                              static_cast<type_real>(perm_det[i]);
      inverse_permzz_val[i] = static_cast<type_real>(permxx[i]) /
                              static_cast<type_real>(perm_det[i]);
      auto phi_over_tort = static_cast<type_real>(phi[i]) /
                           static_cast<type_real>(tortuosity[i]);
      auto afactor =
          rho_bar_val[i] - phi_over_tort * static_cast<type_real>(rho_f[i]);
      vs_expected[i] = Kokkos::sqrt(static_cast<type_real>(mu_G[i]) / afactor);
    }
  } else {
    // Sandstone-like poroelastic material for scalar case
    phi = 0.2;        // porosity
    rho_s = 2650.0;   // solid density (kg/m³)
    rho_f = 1000.0;   // fluid density (kg/m³)
    tortuosity = 2.0; // tortuosity
    mu_G = 10.0e9;    // shear modulus (Pa)
    H_Biot = 25.0e9;  // Biot's H modulus (Pa)
    C_Biot = 10.0e9;  // Biot's C modulus (Pa)
    M_Biot = 15.0e9;  // Biot's M modulus (Pa)
    permxx = 1.0e-12; // permeability in xx (m²)
    permxz = 0.0;     // permeability in xz (m²)
    permzz = 1.0e-12; // permeability in zz (m²)
    eta_f = 1.0e-3;   // fluid viscosity (Pa·s)

    // Compute expected values for verification
    lambda_G_val = H_Biot - 2.0 * mu_G;
    rho_bar_val = (1.0 - phi) * rho_s + phi * rho_f;
    perm_det = permxx * permzz - permxz * permxz;
    inverse_permxx_val = permzz / perm_det;
    inverse_permxz_val = -permxz / perm_det;
    inverse_permzz_val = permxx / perm_det;

    auto phi_over_tort = phi / tortuosity;
    auto afactor = rho_bar_val - phi_over_tort * rho_f;
    vs_expected = Kokkos::sqrt(mu_G / afactor);
  }

  // Create the properties object
  specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic,
      specfem::element::property_tag::isotropic, using_simd>
      props(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot, M_Biot, permxx,
            permxz, permzz, eta_f);

  // Test accessors with tolerance-based comparisons
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.phi() - phi) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_s() - rho_s) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_f() - rho_f) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.tortuosity() - tortuosity) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.mu_G() - mu_G) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.H_Biot() - H_Biot) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.C_Biot() - C_Biot) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.M_Biot() - M_Biot) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permxx() - permxx) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permxz() - permxz) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permzz() - permzz) < tol));
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.eta_f() - eta_f) < tol));

  // Test computed properties
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambda_G() - lambda_G_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambdaplus2mu_G() - H_Biot) < tol));

  // Test inverse permeability calculations
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permxx() - inverse_permxx_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permxz() - inverse_permxz_val) < tol));
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permzz() - inverse_permzz_val) < tol));

  // Test average density calculation
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_bar() - rho_bar_val) < tol));

  // Wave velocities are complex calculations, so we just check they return
  // reasonable values
  EXPECT_TRUE(specfem::datatype::all_of(props.vpI() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(props.vpII() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(props.vs() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(
      props.vpII() < props.vpI())); // vpII is typically slower than vpI
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.vs() - vs_expected) < tol));
}
