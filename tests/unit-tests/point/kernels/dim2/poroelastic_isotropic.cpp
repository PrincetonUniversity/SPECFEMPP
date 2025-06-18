#include "../kernels_tests.hpp"
#include "datatypes/simd.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

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
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, using_simd>
      kernels(rhot, rhof, eta, sm, mu_fr, B, C, M, cpI, cpII, cs, rhobb, rhofbb,
              ratio, phib);

  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.rhot() - rhot) < tol))
      << ExpectedGot(rhot, kernels.rhot());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.rhof() - rhof) < tol))
      << ExpectedGot(rhof, kernels.rhof());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.eta() - eta) < tol))
      << ExpectedGot(eta, kernels.eta());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.sm() - sm) < tol))
      << ExpectedGot(sm, kernels.sm());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.mu_fr() - mu_fr) < tol))
      << ExpectedGot(mu_fr, kernels.mu_fr());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.B() - B) < tol))
      << ExpectedGot(B, kernels.B());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.C() - C) < tol))
      << ExpectedGot(C, kernels.C());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.M() - M) < tol))
      << ExpectedGot(M, kernels.M());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.cpI() - cpI) < tol))
      << ExpectedGot(cpI, kernels.cpI());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.cpII() - cpII) < tol))
      << ExpectedGot(cpII, kernels.cpII());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.cs() - cs) < tol))
      << ExpectedGot(cs, kernels.cs());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.rhobb() - rhobb) < tol))
      << ExpectedGot(rhobb, kernels.rhobb());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.rhofbb() - rhofbb) < tol))
      << ExpectedGot(rhofbb, kernels.rhofbb());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.ratio() - ratio) < tol))
      << ExpectedGot(ratio, kernels.ratio());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.phib() - phib) < tol))
      << ExpectedGot(phib, kernels.phib());

  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(kernels.mu_frb() - expected_mu_frb) < tol))
      << ExpectedGot(expected_mu_frb, kernels.mu_frb());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(kernels.rhob() - expected_rhob) < tol))
      << ExpectedGot(expected_rhob, kernels.rhob());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(kernels.rhofb() - expected_rhofb) < tol))
      << ExpectedGot(expected_rhofb, kernels.rhofb());
  EXPECT_TRUE(specfem::datatype::all_of(
      Kokkos::abs(kernels.phi() - expected_phi) < tol))
      << ExpectedGot(expected_phi, kernels.phi());
}
