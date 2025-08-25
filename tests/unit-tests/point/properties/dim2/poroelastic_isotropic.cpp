#include "../properties_tests.hpp"
#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TYPED_TEST(PointPropertiesTest, PoroelasticIsotropic2D) {
  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
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
    T phi_arr[simd_size];
    T rho_s_arr[simd_size];
    T rho_f_arr[simd_size];
    T tortuosity_arr[simd_size];
    T mu_G_arr[simd_size];
    T H_Biot_arr[simd_size];
    T C_Biot_arr[simd_size];
    T M_Biot_arr[simd_size];
    T permxx_arr[simd_size];
    T permxz_arr[simd_size];
    T permzz_arr[simd_size];
    T eta_f_arr[simd_size];

    T lambda_G_arr[simd_size];
    T rho_bar_arr[simd_size];
    T perm_det_arr[simd_size];
    T inverse_permxx_arr[simd_size];
    T inverse_permxz_arr[simd_size];
    T inverse_permzz_arr[simd_size];
    T vs_expected_arr[simd_size];
    // Setup test data for SIMD
    for (int i = 0; i < simd_size; ++i) {
      const type_real i_real = 1.0;
      phi_arr[i] = static_cast<type_real>(0.2) +
                   i_real * static_cast<type_real>(0.01); // porosity
      rho_s_arr[i] =
          static_cast<type_real>(2650.0) +
          i_real * static_cast<type_real>(10.0); // solid density (kg/m³)
      rho_f_arr[i] =
          static_cast<type_real>(1000.0) +
          i_real * static_cast<type_real>(5.0); // fluid density (kg/m³)
      tortuosity_arr[i] = static_cast<type_real>(2.0) +
                          i_real * static_cast<type_real>(0.1); // tortuosity
      mu_G_arr[i] =
          static_cast<type_real>(10.0e9) +
          i_real * static_cast<type_real>(1.0e8); // shear modulus (Pa)
      H_Biot_arr[i] =
          static_cast<type_real>(25.0e9) +
          i_real * static_cast<type_real>(1.0e8); // Biot's H modulus (Pa)
      C_Biot_arr[i] =
          static_cast<type_real>(10.0e9) +
          i_real * static_cast<type_real>(1.0e7); // Biot's C modulus (Pa)
      M_Biot_arr[i] =
          static_cast<type_real>(15.0e9) +
          i_real * static_cast<type_real>(1.0e7); // Biot's M modulus (Pa)
      permxx_arr[i] =
          static_cast<type_real>(1.0e-12) +
          i_real * static_cast<type_real>(1.0e-14); // permeability in xx (m²)
      permxz_arr[i] = static_cast<type_real>(0.0);  // permeability in xz (m²)
      permzz_arr[i] =
          static_cast<type_real>(1.0e-12) +
          i_real * static_cast<type_real>(1.0e-14); // permeability in zz (m²)
      eta_f_arr[i] =
          static_cast<type_real>(1.0e-3) +
          i_real * static_cast<type_real>(1.0e-5); // fluid viscosity (Pa·s)

      // Compute expected properties for verification
      lambda_G_arr[i] = H_Biot_arr[i] - 2.0 * mu_G_arr[i];
      rho_bar_arr[i] =
          (1.0 - phi_arr[i]) * rho_s_arr[i] + phi_arr[i] * rho_f_arr[i];
      perm_det_arr[i] =
          permxx_arr[i] * permzz_arr[i] - permxz_arr[i] * permxz_arr[i];
      inverse_permxx_arr[i] = permzz_arr[i] / perm_det_arr[i];
      inverse_permxz_arr[i] = -permxz_arr[i] / perm_det_arr[i];
      inverse_permzz_arr[i] = permxx_arr[i] / perm_det_arr[i];

      auto phi_over_tort = phi_arr[i] / tortuosity_arr[i];
      auto afactor = rho_bar_arr[i] - phi_over_tort * rho_f_arr[i];
      vs_expected_arr[i] = Kokkos::sqrt(mu_G_arr[i] / afactor);
    }
    // Copy to SIMD types
    phi.copy_from(phi_arr, Kokkos::Experimental::simd_flag_default);
    rho_s.copy_from(rho_s_arr, Kokkos::Experimental::simd_flag_default);
    rho_f.copy_from(rho_f_arr, Kokkos::Experimental::simd_flag_default);
    tortuosity.copy_from(tortuosity_arr,
                         Kokkos::Experimental::simd_flag_default);
    mu_G.copy_from(mu_G_arr, Kokkos::Experimental::simd_flag_default);
    H_Biot.copy_from(H_Biot_arr, Kokkos::Experimental::simd_flag_default);
    C_Biot.copy_from(C_Biot_arr, Kokkos::Experimental::simd_flag_default);
    M_Biot.copy_from(M_Biot_arr, Kokkos::Experimental::simd_flag_default);
    permxx.copy_from(permxx_arr, Kokkos::Experimental::simd_flag_default);
    permxz.copy_from(permxz_arr, Kokkos::Experimental::simd_flag_default);
    permzz.copy_from(permzz_arr, Kokkos::Experimental::simd_flag_default);
    eta_f.copy_from(eta_f_arr, Kokkos::Experimental::simd_flag_default);

    lambda_G_val.copy_from(lambda_G_arr,
                           Kokkos::Experimental::simd_flag_default);
    rho_bar_val.copy_from(rho_bar_arr, Kokkos::Experimental::simd_flag_default);
    perm_det.copy_from(perm_det_arr, Kokkos::Experimental::simd_flag_default);
    inverse_permxx_val.copy_from(inverse_permxx_arr,
                                 Kokkos::Experimental::simd_flag_default);
    inverse_permxz_val.copy_from(inverse_permxz_arr,
                                 Kokkos::Experimental::simd_flag_default);
    inverse_permzz_val.copy_from(inverse_permzz_arr,
                                 Kokkos::Experimental::simd_flag_default);
    vs_expected.copy_from(vs_expected_arr,
                          Kokkos::Experimental::simd_flag_default);
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
  using PointPropertiesType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointPropertiesType props(phi, rho_s, rho_f, tortuosity, mu_G, H_Biot, C_Biot,
                            M_Biot, permxx, permxz, permzz, eta_f);

  EXPECT_TRUE(specfem::utilities::is_close(props.phi(), phi))
      << ExpectedGot(phi, props.phi());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_s(), rho_s))
      << ExpectedGot(rho_s, props.rho_s());
  EXPECT_TRUE(specfem::utilities::is_close(props.rho_f(), rho_f))
      << ExpectedGot(rho_f, props.rho_f());
  EXPECT_TRUE(specfem::utilities::is_close(props.tortuosity(), tortuosity))
      << ExpectedGot(tortuosity, props.tortuosity());
  EXPECT_TRUE(specfem::utilities::is_close(props.mu_G(), mu_G))
      << ExpectedGot(mu_G, props.mu_G());
  EXPECT_TRUE(specfem::utilities::is_close(props.H_Biot(), H_Biot))
      << ExpectedGot(H_Biot, props.H_Biot());
  EXPECT_TRUE(specfem::utilities::is_close(props.C_Biot(), C_Biot))
      << ExpectedGot(C_Biot, props.C_Biot());
  EXPECT_TRUE(specfem::utilities::is_close(props.M_Biot(), M_Biot))
      << ExpectedGot(M_Biot, props.M_Biot());
  EXPECT_TRUE(specfem::utilities::is_close(props.permxx(), permxx))
      << ExpectedGot(permxx, props.permxx());
  EXPECT_TRUE(specfem::utilities::is_close(props.permxz(), permxz))
      << ExpectedGot(permxz, props.permxz());
  EXPECT_TRUE(specfem::utilities::is_close(props.permzz(), permzz))
      << ExpectedGot(permzz, props.permzz());
  EXPECT_TRUE(specfem::utilities::is_close(props.eta_f(), eta_f))
      << ExpectedGot(eta_f, props.eta_f());

  EXPECT_TRUE(specfem::utilities::is_close(props.lambda_G(), lambda_G_val))
      << ExpectedGot(lambda_G_val, props.lambda_G());
  EXPECT_TRUE(specfem::utilities::is_close(props.lambdaplus2mu_G(), H_Biot))
      << ExpectedGot(H_Biot, props.lambdaplus2mu_G());

  EXPECT_TRUE(
      specfem::utilities::is_close(props.inverse_permxx(), inverse_permxx_val))
      << ExpectedGot(inverse_permxx_val, props.inverse_permxx());
  EXPECT_TRUE(
      specfem::utilities::is_close(props.inverse_permxz(), inverse_permxz_val))
      << ExpectedGot(inverse_permxz_val, props.inverse_permxz());
  EXPECT_TRUE(
      specfem::utilities::is_close(props.inverse_permzz(), inverse_permzz_val))
      << ExpectedGot(inverse_permzz_val, props.inverse_permzz());

  EXPECT_TRUE(specfem::utilities::is_close(props.rho_bar(), rho_bar_val))
      << ExpectedGot(rho_bar_val, props.rho_bar());

  // Wave velocities are complex calculations, so we just check they return
  // reasonable values
  simd_type zero{ static_cast<type_real>(0.0) };
  EXPECT_TRUE(specfem::utilities::is_close(props.vpI(), zero) == false)
      << ExpectedGot(zero, props.vpI());
  EXPECT_TRUE(specfem::utilities::is_close(props.vpII(), zero) == false)
      << ExpectedGot(zero, props.vpII());
  EXPECT_TRUE(specfem::utilities::is_close(props.vs(), zero) == false)
      << ExpectedGot(zero, props.vs());
  EXPECT_TRUE(specfem::utilities::is_close(props.vpII(), props.vpI()) == false)
      << "vpII is typically slower than vpI\n"
      << ExpectedGot(props.vpII(), props.vpI());
  EXPECT_TRUE(specfem::utilities::is_close(props.vs(), vs_expected))
      << ExpectedGot(vs_expected, props.vs());

  // Additional constructors and assignment tests
  PointPropertiesType props2;
  props2.phi() = phi;
  props2.rho_s() = rho_s;
  props2.rho_f() = rho_f;
  props2.tortuosity() = tortuosity;
  props2.mu_G() = mu_G;
  props2.H_Biot() = H_Biot;
  props2.C_Biot() = C_Biot;
  props2.M_Biot() = M_Biot;
  props2.permxx() = permxx;
  props2.permxz() = permxz;
  props2.permzz() = permzz;
  props2.eta_f() = eta_f;

  simd_type data[] = { phi,    rho_s,  rho_f,  tortuosity, mu_G,   H_Biot,
                       C_Biot, M_Biot, permxx, permxz,     permzz, eta_f };
  PointPropertiesType props3(data);

  PointPropertiesType props4(phi);

  EXPECT_TRUE(props2 == props)
      << ExpectedGot(props2.phi(), props.phi())
      << ExpectedGot(props2.rho_s(), props.rho_s())
      << ExpectedGot(props2.rho_f(), props.rho_f())
      << ExpectedGot(props2.tortuosity(), props.tortuosity())
      << ExpectedGot(props2.mu_G(), props.mu_G())
      << ExpectedGot(props2.H_Biot(), props.H_Biot())
      << ExpectedGot(props2.C_Biot(), props.C_Biot())
      << ExpectedGot(props2.M_Biot(), props.M_Biot())
      << ExpectedGot(props2.permxx(), props.permxx())
      << ExpectedGot(props2.permxz(), props.permxz())
      << ExpectedGot(props2.permzz(), props.permzz())
      << ExpectedGot(props2.eta_f(), props.eta_f());

  EXPECT_TRUE(props2 == props3)
      << ExpectedGot(props3.phi(), props2.phi())
      << ExpectedGot(props3.rho_s(), props2.rho_s())
      << ExpectedGot(props3.rho_f(), props2.rho_f())
      << ExpectedGot(props3.tortuosity(), props2.tortuosity())
      << ExpectedGot(props3.mu_G(), props2.mu_G())
      << ExpectedGot(props3.H_Biot(), props2.H_Biot())
      << ExpectedGot(props3.C_Biot(), props2.C_Biot())
      << ExpectedGot(props3.M_Biot(), props2.M_Biot())
      << ExpectedGot(props3.permxx(), props2.permxx())
      << ExpectedGot(props3.permxz(), props2.permxz())
      << ExpectedGot(props3.permzz(), props2.permzz())
      << ExpectedGot(props3.eta_f(), props2.eta_f());

  EXPECT_TRUE(specfem::utilities::is_close(props4.phi(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.rho_s(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.rho_f(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.tortuosity(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.mu_G(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.H_Biot(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.C_Biot(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.M_Biot(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.permxx(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.permxz(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.permzz(), phi));
  EXPECT_TRUE(specfem::utilities::is_close(props4.eta_f(), phi));
}
