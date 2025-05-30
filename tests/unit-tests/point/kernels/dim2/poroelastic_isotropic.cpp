#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TEST_F(PointKernelsTest, PoroelasticIsotropic2D) {
  // Create test kernel values
  constexpr type_real rhot = 2.5;
  constexpr type_real rhof = 3.0;
  constexpr type_real eta = 4.0;
  constexpr type_real sm = 5.0;
  constexpr type_real mu_fr = 6.0;
  constexpr type_real B = 7.0;
  constexpr type_real C = 8.0;
  constexpr type_real M = 9.0;
  constexpr type_real cpI = 10.0;
  constexpr type_real cpII = 11.0;
  constexpr type_real cs = 12.0;
  constexpr type_real rhobb = 13.0;
  constexpr type_real rhofbb = 14.0;
  constexpr type_real ratio = 15.0;
  constexpr type_real phib = 16.0;

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, false>
      kernels(rhot, rhof, eta, sm, mu_fr, B, C, M, cpI, cpII, cs, rhobb, rhofbb,
              ratio, phib);

  // Test accessors
  EXPECT_REAL_EQ(kernels.rhot(), rhot);
  EXPECT_REAL_EQ(kernels.rhof(), rhof);
  EXPECT_REAL_EQ(kernels.eta(), eta);
  EXPECT_REAL_EQ(kernels.sm(), sm);
  EXPECT_REAL_EQ(kernels.mu_fr(), mu_fr);
  EXPECT_REAL_EQ(kernels.B(), B);
  EXPECT_REAL_EQ(kernels.C(), C);
  EXPECT_REAL_EQ(kernels.M(), M);
  EXPECT_REAL_EQ(kernels.cpI(), cpI);
  EXPECT_REAL_EQ(kernels.cpII(), cpII);
  EXPECT_REAL_EQ(kernels.cs(), cs);
  EXPECT_REAL_EQ(kernels.rhobb(), rhobb);
  EXPECT_REAL_EQ(kernels.rhofbb(), rhofbb);
  EXPECT_REAL_EQ(kernels.ratio(), ratio);
  EXPECT_REAL_EQ(kernels.phib(), phib);

  // Test automatically computed values
  EXPECT_REAL_EQ(kernels.mu_frb(), mu_fr);
  EXPECT_REAL_EQ(kernels.rhob(), (rhot + B + mu_fr));
  EXPECT_REAL_EQ(kernels.rhofb(), (rhof + C + M + sm));
  EXPECT_REAL_EQ(kernels.phi(), (static_cast<type_real>(-1.0) * (sm + M)));
}

// Test SIMD version of the kernels
TEST_F(PointKernelsTest, PoroelasticIsotropic2D_SIMD) {
  // Get the SIMD size from the implementation
  using simd_type = typename specfem::datatype::simd<type_real, true>::datatype;
  constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();

  // Skip test if SIMD is not effectively available
  if (simd_size <= 1) {
    GTEST_SKIP() << "SIMD tests skipped because simd_size <= 1";
    return;
  }

  // Create SIMD data objects
  simd_type rhot_simd;
  simd_type rhof_simd;
  simd_type eta_simd;
  simd_type sm_simd;
  simd_type mu_fr_simd;
  simd_type B_simd;
  simd_type C_simd;
  simd_type M_simd;
  simd_type cpI_simd;
  simd_type cpII_simd;
  simd_type cs_simd;
  simd_type rhobb_simd;
  simd_type rhofbb_simd;
  simd_type ratio_simd;
  simd_type phib_simd;

  // Setup test data
  std::vector<type_real> rhot_values(simd_size);
  std::vector<type_real> rhof_values(simd_size);
  std::vector<type_real> eta_values(simd_size);
  std::vector<type_real> sm_values(simd_size);
  std::vector<type_real> mu_fr_values(simd_size);
  std::vector<type_real> B_values(simd_size);
  std::vector<type_real> C_values(simd_size);
  std::vector<type_real> M_values(simd_size);
  std::vector<type_real> cpI_values(simd_size);
  std::vector<type_real> cpII_values(simd_size);
  std::vector<type_real> cs_values(simd_size);
  std::vector<type_real> rhobb_values(simd_size);
  std::vector<type_real> rhofbb_values(simd_size);
  std::vector<type_real> ratio_values(simd_size);
  std::vector<type_real> phib_values(simd_size);

  // Fill vectors with test values
  for (int i = 0; i < simd_size; ++i) {
    rhot_values[i] = 2.5 + i * 0.1;
    rhof_values[i] = 3.0 + i * 0.1;
    eta_values[i] = 4.0 + i * 0.1;
    sm_values[i] = 5.0 + i * 0.1;
    mu_fr_values[i] = 6.0 + i * 0.1;
    B_values[i] = 7.0 + i * 0.1;
    C_values[i] = 8.0 + i * 0.1;
    M_values[i] = 9.0 + i * 0.1;
    cpI_values[i] = 10.0 + i * 0.1;
    cpII_values[i] = 11.0 + i * 0.1;
    cs_values[i] = 12.0 + i * 0.1;
    rhobb_values[i] = 13.0 + i * 0.1;
    rhofbb_values[i] = 14.0 + i * 0.1;
    ratio_values[i] = 15.0 + i * 0.1;
    phib_values[i] = 16.0 + i * 0.1;

    // Load into SIMD vectors using operator[]
    rhot_simd[i] = rhot_values[i];
    rhof_simd[i] = rhof_values[i];
    eta_simd[i] = eta_values[i];
    sm_simd[i] = sm_values[i];
    mu_fr_simd[i] = mu_fr_values[i];
    B_simd[i] = B_values[i];
    C_simd[i] = C_values[i];
    M_simd[i] = M_values[i];
    cpI_simd[i] = cpI_values[i];
    cpII_simd[i] = cpII_values[i];
    cs_simd[i] = cs_values[i];
    rhobb_simd[i] = rhobb_values[i];
    rhofbb_simd[i] = rhofbb_values[i];
    ratio_simd[i] = ratio_values[i];
    phib_simd[i] = phib_values[i];
  }

  // Create the kernels object with SIMD data
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, true>
      kernels(rhot_simd, rhof_simd, eta_simd, sm_simd, mu_fr_simd, B_simd,
              C_simd, M_simd, cpI_simd, cpII_simd, cs_simd, rhobb_simd,
              rhofbb_simd, ratio_simd, phib_simd);

  // Test that each lane contains the expected values
  for (int lane = 0; lane < simd_size; ++lane) {
    type_real rhot = kernels.rhot()[lane];
    type_real rhof = kernels.rhof()[lane];
    type_real eta = kernels.eta()[lane];
    type_real sm = kernels.sm()[lane];
    type_real mu_fr = kernels.mu_fr()[lane];
    type_real B = kernels.B()[lane];
    type_real C = kernels.C()[lane];
    type_real M = kernels.M()[lane];
    type_real cpI = kernels.cpI()[lane];
    type_real cpII = kernels.cpII()[lane];
    type_real cs = kernels.cs()[lane];
    type_real rhobb = kernels.rhobb()[lane];
    type_real rhofbb = kernels.rhofbb()[lane];
    type_real ratio = kernels.ratio()[lane];
    type_real phib = kernels.phib()[lane];

    // Check derived values
    type_real mu_frb = kernels.mu_frb()[lane];
    type_real rhob = kernels.rhob()[lane];
    type_real rhofb = kernels.rhofb()[lane];
    type_real phi = kernels.phi()[lane];

    EXPECT_REAL_EQ(rhot, rhot_values[lane]);
    EXPECT_REAL_EQ(rhof, rhof_values[lane]);
    EXPECT_REAL_EQ(eta, eta_values[lane]);
    EXPECT_REAL_EQ(sm, sm_values[lane]);
    EXPECT_REAL_EQ(mu_fr, mu_fr_values[lane]);
    EXPECT_REAL_EQ(B, B_values[lane]);
    EXPECT_REAL_EQ(C, C_values[lane]);
    EXPECT_REAL_EQ(M, M_values[lane]);
    EXPECT_REAL_EQ(cpI, cpI_values[lane]);
    EXPECT_REAL_EQ(cpII, cpII_values[lane]);
    EXPECT_REAL_EQ(cs, cs_values[lane]);
    EXPECT_REAL_EQ(rhobb, rhobb_values[lane]);
    EXPECT_REAL_EQ(rhofbb, rhofbb_values[lane]);
    EXPECT_REAL_EQ(ratio, ratio_values[lane]);
    EXPECT_REAL_EQ(phib, phib_values[lane]);

    // Check derived values
    EXPECT_REAL_EQ(mu_frb, mu_fr_values[lane]);
    EXPECT_REAL_EQ(rhob,
                   (rhot_values[lane] + B_values[lane] + mu_fr_values[lane]));
    EXPECT_REAL_EQ(rhofb, (rhof_values[lane] + C_values[lane] + M_values[lane] +
                           sm_values[lane]));
    EXPECT_REAL_EQ(phi, (static_cast<type_real>(-1.0) *
                         (sm_values[lane] + M_values[lane])));
  }
}
