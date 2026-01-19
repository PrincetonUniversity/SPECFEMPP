#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Poroelastic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, PoroelasticIsotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold kernel values
  simd_type rhot;
  simd_type rhof;
  simd_type eta;
  simd_type sm;
  simd_type mu_fr;
  simd_type B;
  simd_type C;
  simd_type M;
  simd_type cpI;
  simd_type cpII;
  simd_type cs;
  simd_type rhobb;
  simd_type rhofbb;
  simd_type ratio;
  simd_type phib;
  simd_type expected_mu_frb;
  simd_type expected_rhob;
  simd_type expected_rhofb;
  simd_type expected_phi;

  if constexpr (using_simd) {
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rhot[i] = static_cast<type_real>(2.5) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);
      rhof[i] = static_cast<type_real>(3.0) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);
      eta[i] = static_cast<type_real>(4.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      sm[i] = static_cast<type_real>(5.0) +
              static_cast<type_real>(i) * static_cast<type_real>(0.1);
      mu_fr[i] = static_cast<type_real>(6.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      B[i] = static_cast<type_real>(7.0) +
             static_cast<type_real>(i) * static_cast<type_real>(0.1);
      C[i] = static_cast<type_real>(8.0) +
             static_cast<type_real>(i) * static_cast<type_real>(0.1);
      M[i] = static_cast<type_real>(9.0) +
             static_cast<type_real>(i) * static_cast<type_real>(0.1);
      cpI[i] = static_cast<type_real>(10.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      cpII[i] = static_cast<type_real>(11.0) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);
      cs[i] = static_cast<type_real>(12.0) +
              static_cast<type_real>(i) * static_cast<type_real>(0.1);
      rhobb[i] = static_cast<type_real>(13.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      rhofbb[i] = static_cast<type_real>(14.0) +
                  static_cast<type_real>(i) * static_cast<type_real>(0.1);
      ratio[i] = static_cast<type_real>(15.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      phib[i] = static_cast<type_real>(16.0) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);

      expected_mu_frb[i] = static_cast<type_real>(mu_fr[i]);
      expected_rhob[i] = static_cast<type_real>(rhot[i]) +
                         static_cast<type_real>(B[i]) +
                         static_cast<type_real>(mu_fr[i]);
      expected_rhofb[i] =
          static_cast<type_real>(rhof[i]) + static_cast<type_real>(C[i]) +
          static_cast<type_real>(M[i]) + static_cast<type_real>(sm[i]);
      expected_phi[i] =
          static_cast<type_real>(-1.0) *
          (static_cast<type_real>(sm[i]) + static_cast<type_real>(M[i]));
    }
  } else {
    // For scalar case, we need direct assignment
    rhot = static_cast<type_real>(2.5);
    rhof = static_cast<type_real>(3.0);
    eta = static_cast<type_real>(4.0);
    sm = static_cast<type_real>(5.0);
    mu_fr = static_cast<type_real>(6.0);
    B = static_cast<type_real>(7.0);
    C = static_cast<type_real>(8.0);
    M = static_cast<type_real>(9.0);
    cpI = static_cast<type_real>(10.0);
    cpII = static_cast<type_real>(11.0);
    cs = static_cast<type_real>(12.0);
    rhobb = static_cast<type_real>(13.0);
    rhofbb = static_cast<type_real>(14.0);
    ratio = static_cast<type_real>(15.0);
    phib = static_cast<type_real>(16.0);

    expected_mu_frb = mu_fr;
    expected_rhob = rhot + B + mu_fr;
    expected_rhofb = rhof + C + M + sm;
    expected_phi = static_cast<type_real>(-1.0) * (sm + M);
  }

  // Create the kernels object
  using PointKernelType = specfem::point::kernels<
      specfem::dimension::type::dim2, specfem::element::medium_tag::poroelastic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointKernelType kernels(rhot, rhof, eta, sm, mu_fr, B, C, M, cpI, cpII, cs,
                          rhobb, rhofbb, ratio, phib);

  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhot(), rhot))
      << ExpectedGot(rhot, kernels.rhot());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhof(), rhof))
      << ExpectedGot(rhof, kernels.rhof());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.eta(), eta))
      << ExpectedGot(eta, kernels.eta());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.sm(), sm))
      << ExpectedGot(sm, kernels.sm());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.mu_fr(), mu_fr))
      << ExpectedGot(mu_fr, kernels.mu_fr());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.B(), B))
      << ExpectedGot(B, kernels.B());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.C(), C))
      << ExpectedGot(C, kernels.C());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.M(), M))
      << ExpectedGot(M, kernels.M());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.cpI(), cpI))
      << ExpectedGot(cpI, kernels.cpI());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.cpII(), cpII))
      << ExpectedGot(cpII, kernels.cpII());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.cs(), cs))
      << ExpectedGot(cs, kernels.cs());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhobb(), rhobb))
      << ExpectedGot(rhobb, kernels.rhobb());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhofbb(), rhofbb))
      << ExpectedGot(rhofbb, kernels.rhofbb());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.ratio(), ratio))
      << ExpectedGot(ratio, kernels.ratio());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.phib(), phib))
      << ExpectedGot(phib, kernels.phib());

  EXPECT_TRUE(specfem::utilities::is_close(kernels.mu_frb(), expected_mu_frb))
      << ExpectedGot(expected_mu_frb, kernels.mu_frb());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhob(), expected_rhob))
      << ExpectedGot(expected_rhob, kernels.rhob());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhofb(), expected_rhofb))
      << ExpectedGot(expected_rhofb, kernels.rhofb());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.phi(), expected_phi))
      << ExpectedGot(expected_phi, kernels.phi());

  PointKernelType kernels2;
  kernels2.rhot() = rhot;
  kernels2.rhof() = rhof;
  kernels2.eta() = eta;
  kernels2.sm() = sm;
  kernels2.mu_fr() = mu_fr;
  kernels2.B() = B;
  kernels2.C() = C;
  kernels2.M() = M;
  kernels2.cpI() = cpI;
  kernels2.cpII() = cpII;
  kernels2.cs() = cs;
  kernels2.rhobb() = rhobb;
  kernels2.rhofbb() = rhofbb;
  kernels2.ratio() = ratio;
  kernels2.phib() = phib;
  kernels2.mu_frb() = expected_mu_frb;
  kernels2.rhob() = expected_rhob;
  kernels2.rhofb() = expected_rhofb;
  kernels2.phi() = expected_phi;

  simd_type data[] = { rhot,
                       rhof,
                       eta,
                       sm,
                       mu_fr,
                       B,
                       C,
                       M,
                       mu_fr,
                       expected_rhob,
                       expected_rhofb,
                       expected_phi,
                       cpI,
                       cpII,
                       cs,
                       rhobb,
                       rhofbb,
                       ratio,
                       phib };
  PointKernelType kernels3(data);

  PointKernelType kernels4(rhot);

  EXPECT_TRUE(kernels == kernels2)
      << ExpectedGot(kernels2.rhot(), kernels.rhot())
      << ExpectedGot(kernels2.rhof(), kernels.rhof())
      << ExpectedGot(kernels2.eta(), kernels.eta())
      << ExpectedGot(kernels2.sm(), kernels.sm())
      << ExpectedGot(kernels2.mu_fr(), kernels.mu_fr())
      << ExpectedGot(kernels2.B(), kernels.B())
      << ExpectedGot(kernels2.C(), kernels.C())
      << ExpectedGot(kernels2.M(), kernels.M())
      << ExpectedGot(kernels2.cpI(), kernels.cpI())
      << ExpectedGot(kernels2.cpII(), kernels.cpII())
      << ExpectedGot(kernels2.cs(), kernels.cs())
      << ExpectedGot(kernels2.rhobb(), kernels.rhobb())
      << ExpectedGot(kernels2.rhofbb(), kernels.rhofbb())
      << ExpectedGot(kernels2.ratio(), kernels.ratio())
      << ExpectedGot(kernels2.phib(), kernels.phib());

  EXPECT_TRUE(kernels2 == kernels3)
      << ExpectedGot(kernels3.rhot(), kernels2.rhot())
      << ExpectedGot(kernels3.rhof(), kernels2.rhof())
      << ExpectedGot(kernels3.eta(), kernels2.eta())
      << ExpectedGot(kernels3.sm(), kernels2.sm())
      << ExpectedGot(kernels3.mu_fr(), kernels2.mu_fr())
      << ExpectedGot(kernels3.B(), kernels2.B())
      << ExpectedGot(kernels3.C(), kernels2.C())
      << ExpectedGot(kernels3.M(), kernels2.M())
      << ExpectedGot(kernels3.cpI(), kernels2.cpI())
      << ExpectedGot(kernels3.cpII(), kernels2.cpII())
      << ExpectedGot(kernels3.cs(), kernels2.cs())
      << ExpectedGot(kernels3.rhobb(), kernels2.rhobb())
      << ExpectedGot(kernels3.rhofbb(), kernels2.rhofbb())
      << ExpectedGot(kernels3.ratio(), kernels2.ratio())
      << ExpectedGot(kernels3.phib(), kernels2.phib());

  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rhot(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rhof(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.eta(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.sm(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.mu_fr(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.B(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.C(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.M(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.cpI(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.cpII(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.cs(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rhobb(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rhofbb(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.ratio(), rhot));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.phib(), rhot));
}
