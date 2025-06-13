#include "../properties_tests.hpp"
#include "datatypes/simd.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, PoroelasticIsotropic2D) {
  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

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

  if constexpr (using_simd) {
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      phi[i] =
          static_cast<type_real>(0.2) +
          static_cast<type_real>(i) * static_cast<type_real>(0.01); // porosity
      rho_s[i] = static_cast<type_real>(2650.0) +
                 static_cast<type_real>(i) *
                     static_cast<type_real>(10.0); // solid density (kg/m³)
      rho_f[i] = static_cast<type_real>(1000.0) +
                 static_cast<type_real>(i) *
                     static_cast<type_real>(5.0); // fluid density (kg/m³)
      tortuosity[i] =
          static_cast<type_real>(2.0) +
          static_cast<type_real>(i) * static_cast<type_real>(0.1); // tortuosity
      mu_G[i] = static_cast<type_real>(10.0e9) +
                static_cast<type_real>(i) *
                    static_cast<type_real>(1.0e8); // shear modulus (Pa)
      H_Biot[i] = static_cast<type_real>(25.0e9) +
                  static_cast<type_real>(i) *
                      static_cast<type_real>(1.0e8); // Biot's H modulus (Pa)
      C_Biot[i] = static_cast<type_real>(10.0e9) +
                  static_cast<type_real>(i) *
                      static_cast<type_real>(1.0e7); // Biot's C modulus (Pa)
      M_Biot[i] = static_cast<type_real>(15.0e9) +
                  static_cast<type_real>(i) *
                      static_cast<type_real>(1.0e7); // Biot's M modulus (Pa)
      permxx[i] =
          static_cast<type_real>(1.0e-12) +
          static_cast<type_real>(i) *
              static_cast<type_real>(1.0e-14); // permeability in xx (m²)
      permxz[i] = static_cast<type_real>(0.0); // permeability in xz (m²)
      permzz[i] =
          static_cast<type_real>(1.0e-12) +
          static_cast<type_real>(i) *
              static_cast<type_real>(1.0e-14); // permeability in zz (m²)
      eta_f[i] = static_cast<type_real>(1.0e-3) +
                 static_cast<type_real>(i) *
                     static_cast<type_real>(1.0e-5); // fluid viscosity (Pa·s)

      // Compute expected properties for verification
      lambda_G_val[i] =
          static_cast<type_real>(H_Biot[i]) -
          static_cast<type_real>(2.0) * static_cast<type_real>(mu_G[i]);
      rho_bar_val[i] =
          (static_cast<type_real>(1.0) - static_cast<type_real>(phi[i])) *
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
    lambda_G_val = H_Biot - static_cast<type_real>(2.0) * mu_G;
    rho_bar_val = (static_cast<type_real>(1.0) - phi) * rho_s + phi * rho_f;
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

  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.phi() - phi) < tol))
      << ExpectedGot(phi, props.phi());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_s() - rho_s) < tol))
      << ExpectedGot(rho_s, props.rho_s());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.rho_f() - rho_f) < tol))
      << ExpectedGot(rho_f, props.rho_f());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.tortuosity() - tortuosity) < tol))
      << ExpectedGot(tortuosity, props.tortuosity());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(props.mu_G() - mu_G) < tol))
      << ExpectedGot(mu_G, props.mu_G());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.H_Biot() - H_Biot) < tol))
      << ExpectedGot(H_Biot, props.H_Biot());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.C_Biot() - C_Biot) < tol))
      << ExpectedGot(C_Biot, props.C_Biot());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.M_Biot() - M_Biot) < tol))
      << ExpectedGot(M_Biot, props.M_Biot());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permxx() - permxx) < tol))
      << ExpectedGot(permxx, props.permxx());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permxz() - permxz) < tol))
      << ExpectedGot(permxz, props.permxz());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.permzz() - permzz) < tol))
      << ExpectedGot(permzz, props.permzz());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.eta_f() - eta_f) < tol))
      << ExpectedGot(eta_f, props.eta_f());

  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambda_G() - lambda_G_val) < tol))
      << ExpectedGot(lambda_G_val, props.lambda_G());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.lambdaplus2mu_G() - H_Biot) < tol))
      << ExpectedGot(H_Biot, props.lambdaplus2mu_G());

  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permxx() - inverse_permxx_val) < tol))
      << ExpectedGot(inverse_permxx_val, props.inverse_permxx());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permxz() - inverse_permxz_val) < tol))
      << ExpectedGot(inverse_permxz_val, props.inverse_permxz());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.inverse_permzz() - inverse_permzz_val) < tol))
      << ExpectedGot(inverse_permzz_val, props.inverse_permzz());

  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(props.rho_bar() - rho_bar_val) < tol))
      << ExpectedGot(rho_bar_val, props.rho_bar());

  // Wave velocities are complex calculations, so we just check they return
  // reasonable values
  EXPECT_TRUE(specfem::datatype::all_of(props.vpI() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(props.vpII() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(props.vs() > 0.0));
  EXPECT_TRUE(specfem::datatype::all_of(
      props.vpII() < props.vpI())); // vpII is typically slower than vpI
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(props.vs() - vs_expected) < tol))
      << ExpectedGot(vs_expected, props.vs());
}
