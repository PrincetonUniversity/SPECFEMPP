#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TEST_F(PointPropertiesTest, PoroelasticIsotropic2D) {
  // Sandstone-like poroelastic material
  constexpr type_real phi = 0.2;        // porosity
  constexpr type_real rho_s = 2650.0;   // solid density (kg/m³)
  constexpr type_real rho_f = 1000.0;   // fluid density (kg/m³)
  constexpr type_real tortuosity = 2.0; // tortuosity
  constexpr type_real mu_G = 10.0e9;    // shear modulus (Pa)
  constexpr type_real H_Biot = 25.0e9;  // Biot's H modulus (Pa)
  constexpr type_real C_Biot = 10.0e9;  // Biot's C modulus (Pa)
  constexpr type_real M_Biot = 15.0e9;  // Biot's M modulus (Pa)
  constexpr type_real permxx = 1.0e-12; // permeability in xx (m²)
  constexpr type_real permxz = 0.0;     // permeability in xz (m²)
  constexpr type_real permzz = 1.0e-12; // permeability in zz (m²)
  constexpr type_real eta_f = 1.0e-3;   // fluid viscosity (Pa·s)

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::poroelastic,
                             specfem::element::property_tag::isotropic, false>
      props(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot, M_Biot, permxx,
            permxz, permzz, eta_f);

  // Test accessors
  EXPECT_REAL_EQ(props.phi(), phi);
  EXPECT_REAL_EQ(props.rho_s(), rho_s);
  EXPECT_REAL_EQ(props.rho_f(), rho_f);
  EXPECT_REAL_EQ(props.tortuosity(), tortuosity);
  EXPECT_REAL_EQ(props.mu_G(), mu_G);
  EXPECT_REAL_EQ(props.H_Biot(), H_Biot);
  EXPECT_REAL_EQ(props.C_Biot(), C_Biot);
  EXPECT_REAL_EQ(props.M_Biot(), M_Biot);
  EXPECT_REAL_EQ(props.permxx(), permxx);
  EXPECT_REAL_EQ(props.permxz(), permxz);
  EXPECT_REAL_EQ(props.permzz(), permzz);
  EXPECT_REAL_EQ(props.eta_f(), eta_f);

  // Test computed properties
  constexpr type_real lambda_G = H_Biot - 2.0 * mu_G;
  EXPECT_REAL_EQ(props.lambda_G(), lambda_G);
  EXPECT_REAL_EQ(props.lambdaplus2mu_G(), H_Biot);

  // Test inverse permeability calculations
  const type_real determinant = permxx * permzz - permxz * permxz;
  EXPECT_REAL_EQ(props.inverse_permxx(), permzz / determinant);
  EXPECT_REAL_EQ(props.inverse_permxz(), -permxz / determinant);
  EXPECT_REAL_EQ(props.inverse_permzz(), permxx / determinant);

  // Test average density calculation
  const type_real rho_bar = (1.0 - phi) * rho_s + phi * rho_f;
  EXPECT_REAL_EQ(props.rho_bar(), rho_bar);

  // Wave velocities are complex calculations, so we just check they return
  // reasonable values
  EXPECT_GT(props.vpI(), 0.0);
  EXPECT_GT(props.vpII(), 0.0);
  EXPECT_GT(props.vs(), 0.0);
  EXPECT_LT(props.vpII(), props.vpI()); // vpII is typically slower than vpI
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, PoroelasticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type phi_simd;
  simd_type rho_s_simd;
  simd_type rho_f_simd;
  simd_type tortuosity_simd;
  simd_type mu_G_simd;
  simd_type H_Biot_simd;
  simd_type C_Biot_simd;
  simd_type M_Biot_simd;
  simd_type permxx_simd;
  simd_type permxz_simd;
  simd_type permzz_simd;
  simd_type eta_f_simd;

  // Setup test data
  std::vector<type_real> phi_values(simd_size);
  std::vector<type_real> rho_s_values(simd_size);
  std::vector<type_real> rho_f_values(simd_size);
  std::vector<type_real> tortuosity_values(simd_size);
  std::vector<type_real> mu_G_values(simd_size);
  std::vector<type_real> H_Biot_values(simd_size);
  std::vector<type_real> C_Biot_values(simd_size);
  std::vector<type_real> M_Biot_values(simd_size);
  std::vector<type_real> permxx_values(simd_size);
  std::vector<type_real> permxz_values(simd_size);
  std::vector<type_real> permzz_values(simd_size);
  std::vector<type_real> eta_f_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    phi_values[i] = static_cast<type_real>(0.2 + i * 0.01); // porosity
    rho_s_values[i] =
        static_cast<type_real>(2650.0 + i * 10.0); // solid density (kg/m³)
    rho_f_values[i] =
        static_cast<type_real>(1000.0 + i * 5.0); // fluid density (kg/m³)
    tortuosity_values[i] = static_cast<type_real>(2.0 + i * 0.1); // tortuosity
    mu_G_values[i] =
        static_cast<type_real>(10.0e9 + i * 1.0e8); // shear modulus (Pa)
    H_Biot_values[i] =
        static_cast<type_real>(25.0e9 + i * 1.0e8); // Biot's H modulus (Pa)
    C_Biot_values[i] =
        static_cast<type_real>(10.0e9 + i * 1.0e7); // Biot's C modulus (Pa)
    M_Biot_values[i] =
        static_cast<type_real>(15.0e9 + i * 1.0e7); // Biot's M modulus (Pa)
    permxx_values[i] = static_cast<type_real>(
        1.0e-12 + i * 1.0e-14);                     // permeability in xx (m²)
    permxz_values[i] = static_cast<type_real>(0.0); // permeability in xz (m²)
    permzz_values[i] = static_cast<type_real>(
        1.0e-12 + i * 1.0e-14); // permeability in zz (m²)
    eta_f_values[i] =
        static_cast<type_real>(1.0e-3 + i * 1.0e-5); // fluid viscosity (Pa·s)

    // Load into SIMD vectors using operator[]
    phi_simd[i] = phi_values[i];
    rho_s_simd[i] = rho_s_values[i];
    rho_f_simd[i] = rho_f_values[i];
    tortuosity_simd[i] = tortuosity_values[i];
    mu_G_simd[i] = mu_G_values[i];
    H_Biot_simd[i] = H_Biot_values[i];
    C_Biot_simd[i] = C_Biot_values[i];
    M_Biot_simd[i] = M_Biot_values[i];
    permxx_simd[i] = permxx_values[i];
    permxz_simd[i] = permxz_values[i];
    permzz_simd[i] = permzz_values[i];
    eta_f_simd[i] = eta_f_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::poroelastic,
                             specfem::element::property_tag::isotropic, true>
      props(phi_simd, rho_s_simd, rho_f_simd, tortuosity_simd, mu_G_simd,
            H_Biot_simd, C_Biot_simd, M_Biot_simd, permxx_simd, permxz_simd,
            permzz_simd, eta_f_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real phi = props.phi()[lane];
    type_real rho_s = props.rho_s()[lane];
    type_real rho_f = props.rho_f()[lane];
    type_real tortuosity = props.tortuosity()[lane];
    type_real mu_G = props.mu_G()[lane];
    type_real H_Biot = props.H_Biot()[lane];
    type_real C_Biot = props.C_Biot()[lane];
    type_real M_Biot = props.M_Biot()[lane];
    type_real permxx = props.permxx()[lane];
    type_real permxz = props.permxz()[lane];
    type_real permzz = props.permzz()[lane];
    type_real eta_f = props.eta_f()[lane];

    EXPECT_REAL_EQ(phi, phi_values[lane]);
    EXPECT_REAL_EQ(rho_s, rho_s_values[lane]);
    EXPECT_REAL_EQ(rho_f, rho_f_values[lane]);
    EXPECT_REAL_EQ(tortuosity, tortuosity_values[lane]);
    EXPECT_REAL_EQ(mu_G, mu_G_values[lane]);
    EXPECT_REAL_EQ(H_Biot, H_Biot_values[lane]);
    EXPECT_REAL_EQ(C_Biot, C_Biot_values[lane]);
    EXPECT_REAL_EQ(M_Biot, M_Biot_values[lane]);
    EXPECT_REAL_EQ(permxx, permxx_values[lane]);
    EXPECT_REAL_EQ(permxz, permxz_values[lane]);
    EXPECT_REAL_EQ(permzz, permzz_values[lane]);
    EXPECT_REAL_EQ(eta_f, eta_f_values[lane]);

    // Test computed properties
    type_real lambda_G = H_Biot_values[lane] - 2.0 * mu_G_values[lane];
    EXPECT_REAL_EQ(props.lambda_G()[lane], lambda_G);
    EXPECT_REAL_EQ(props.lambdaplus2mu_G()[lane], H_Biot_values[lane]);

    // Calculate rho_bar (average density)
    type_real rho_bar = (1.0 - phi_values[lane]) * rho_s_values[lane] +
                        phi_values[lane] * rho_f_values[lane];
    EXPECT_REAL_EQ(props.rho_bar()[lane], rho_bar);

    // For poroelastic wave velocities, we can't directly compare with simple
    // formulas So we'll check they return reasonable values
    EXPECT_GT(props.vpI()[lane], 0.0);
    EXPECT_GT(props.vpII()[lane], 0.0);
    EXPECT_GT(props.vs()[lane], 0.0);

    // For shear wave velocity, validate against the formula in properties.hpp
    type_real phi_over_tort = phi_values[lane] / tortuosity_values[lane];
    type_real afactor = rho_bar - phi_over_tort * rho_f_values[lane];
    type_real vs_expected = std::sqrt(mu_G_values[lane] / afactor);
    EXPECT_REAL_EQ(props.vs()[lane], vs_expected);

    // Test that vpII is slower than vpI (physical expectation)
    EXPECT_LT(props.vpII()[lane], props.vpI()[lane]);

    // Calculate permeability determinant
    type_real perm_det = permxx_values[lane] * permzz_values[lane] -
                         permxz_values[lane] * permxz_values[lane];

    // Test inverse permeabilities
    EXPECT_REAL_EQ(props.inverse_permxx()[lane],
                   permzz_values[lane] / perm_det);
    EXPECT_REAL_EQ(props.inverse_permxz()[lane],
                   -permxz_values[lane] / perm_det);
    EXPECT_REAL_EQ(props.inverse_permzz()[lane],
                   permxx_values[lane] / perm_det);
  }
}
