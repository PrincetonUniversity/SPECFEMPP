#include "specfem/point/properties.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Test fixture for point properties tests
class PointPropertiesTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if needed for tests
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos if needed
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};

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

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticIsotropic2D) {
  // Granite-like material
  constexpr type_real rho = 2700.0;                    // kg/m³
  constexpr type_real vp = 6000.0;                     // m/s
  constexpr type_real vs = 3500.0;                     // m/s
  constexpr type_real mu = rho * vs * vs;              // shear modulus
  constexpr type_real lambda = rho * vp * vp - 2 * mu; // first Lamé parameter
  constexpr type_real lambdaplus2mu = lambda + 2 * mu;

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, false>
      props(lambdaplus2mu, mu, rho);

  // Test accessors
  EXPECT_REAL_EQ(props.lambdaplus2mu(), lambdaplus2mu);
  EXPECT_REAL_EQ(props.mu(), mu);
  EXPECT_REAL_EQ(props.rho(), rho);

  // Test computed properties
  EXPECT_REAL_EQ(props.lambda(), lambda);
  EXPECT_REAL_EQ(props.rho_vp(), rho * vp);
  EXPECT_REAL_EQ(props.rho_vs(), rho * vs);
}

// Test SIMD version of the properties
TEST_F(PointPropertiesTest, ElasticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type lambdaplus2mu_simd;
  simd_type mu_simd;
  simd_type rho_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> vp_values(simd_size);
  std::vector<type_real> vs_values(simd_size);
  std::vector<type_real> mu_values(simd_size);
  std::vector<type_real> lambda_values(simd_size);
  std::vector<type_real> lambdaplus2mu_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2700.0 + i * 50.0; // kg/m³
    vp_values[i] = 6000.0 + i * 100.0; // m/s
    vs_values[i] = 3500.0 + i * 50.0;  // m/s
    mu_values[i] = rho_values[i] * vs_values[i] * vs_values[i];
    lambda_values[i] =
        rho_values[i] * vp_values[i] * vp_values[i] - 2.0 * mu_values[i];
    lambdaplus2mu_values[i] = lambda_values[i] + 2.0 * mu_values[i];

    // Load into SIMD vectors
    lambdaplus2mu_simd[i] = lambdaplus2mu_values[i];
    mu_simd[i] = mu_values[i];
    rho_simd[i] = rho_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim2,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, true>
      props(lambdaplus2mu_simd, mu_simd, rho_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real lambdaplus2mu = props.lambdaplus2mu()[lane];
    type_real mu = props.mu()[lane];
    type_real rho = props.rho()[lane];
    type_real lambda = props.lambda()[lane];
    type_real rho_vp = props.rho_vp()[lane];
    type_real rho_vs = props.rho_vs()[lane];

    EXPECT_REAL_EQ(lambdaplus2mu, lambdaplus2mu_values[lane]);
    EXPECT_REAL_EQ(mu, mu_values[lane]);
    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(lambda, lambda_values[lane]);
    EXPECT_REAL_EQ(rho_vp, rho_values[lane] * vp_values[lane]);
    EXPECT_REAL_EQ(rho_vs, rho_values[lane] * vs_values[lane]);
  }
}

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

// ============================================================================
// 3D Elastic Tests
// ============================================================================
TEST_F(PointPropertiesTest, ElasticIsotropic3D) {
  // Granite-like material
  constexpr type_real rho = 2700.0;       // kg/m³
  constexpr type_real vp = 6000.0;        // m/s
  constexpr type_real vs = 3500.0;        // m/s
  constexpr type_real mu = rho * vs * vs; // shear modulus

  // For 3D, we use kappa (bulk modulus) instead of lambda + 2*mu
  constexpr type_real kappa =
      rho * (vp * vp - (4.0 / 3.0) * vs * vs); // bulk modulus
  constexpr type_real lambdaplus2mu = kappa + (4.0 / 3.0) * mu;
  constexpr type_real lambda = lambdaplus2mu - 2.0 * mu;

  // Create the properties object
  specfem::point::properties<specfem::dimension::type::dim3,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, false>
      props(kappa, mu, rho);

  // Test accessors
  EXPECT_REAL_EQ(props.kappa(), kappa);
  EXPECT_REAL_EQ(props.mu(), mu);
  EXPECT_REAL_EQ(props.rho(), rho);

  // Test computed properties
  EXPECT_REAL_EQ(props.lambdaplus2mu(), lambdaplus2mu);
  EXPECT_REAL_EQ(props.lambda(), lambda);
  EXPECT_REAL_EQ(props.rho_vp(), rho * vp);
  EXPECT_REAL_EQ(props.rho_vs(), rho * vs);
}

TEST_F(PointPropertiesTest, ElasticIsotropic3D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type kappa_simd;
  simd_type mu_simd;
  simd_type rho_simd;

  // Setup test data
  std::vector<type_real> rho_values(simd_size);
  std::vector<type_real> vp_values(simd_size);
  std::vector<type_real> vs_values(simd_size);
  std::vector<type_real> mu_values(simd_size);
  std::vector<type_real> kappa_values(simd_size);
  std::vector<type_real> lambdaplus2mu_values(simd_size);
  std::vector<type_real> lambda_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rho_values[i] = 2700.0 + i * 50.0; // kg/m³
    vp_values[i] = 6000.0 + i * 100.0; // m/s
    vs_values[i] = 3500.0 + i * 50.0;  // m/s
    mu_values[i] = rho_values[i] * vs_values[i] * vs_values[i];
    kappa_values[i] =
        rho_values[i] * (vp_values[i] * vp_values[i] -
                         (4.0 / 3.0) * vs_values[i] * vs_values[i]);
    lambdaplus2mu_values[i] = kappa_values[i] + (4.0 / 3.0) * mu_values[i];
    lambda_values[i] = lambdaplus2mu_values[i] - 2.0 * mu_values[i];

    // Load into SIMD vectors
    kappa_simd[i] = kappa_values[i];
    mu_simd[i] = mu_values[i];
    rho_simd[i] = rho_values[i];
  }

  // Create the properties object with SIMD data
  specfem::point::properties<specfem::dimension::type::dim3,
                             specfem::element::medium_tag::elastic,
                             specfem::element::property_tag::isotropic, true>
      props(kappa_simd, mu_simd, rho_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real kappa = props.kappa()[lane];
    type_real mu = props.mu()[lane];
    type_real rho = props.rho()[lane];
    type_real lambdaplus2mu = props.lambdaplus2mu()[lane];
    type_real lambda = props.lambda()[lane];
    type_real rho_vp = props.rho_vp()[lane];
    type_real rho_vs = props.rho_vs()[lane];

    EXPECT_REAL_EQ(kappa, kappa_values[lane]);
    EXPECT_REAL_EQ(mu, mu_values[lane]);
    EXPECT_REAL_EQ(rho, rho_values[lane]);
    EXPECT_REAL_EQ(lambdaplus2mu, lambdaplus2mu_values[lane]);
    EXPECT_REAL_EQ(lambda, lambda_values[lane]);
    EXPECT_REAL_EQ(rho_vp, rho_values[lane] * vp_values[lane]);
    EXPECT_REAL_EQ(rho_vs, rho_values[lane] * vs_values[lane]);
  }
}

// ============================================================================
// SIMD Tests
// ============================================================================

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
