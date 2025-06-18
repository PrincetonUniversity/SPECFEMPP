#include "../kernels_tests.hpp"
#include "datatypes/simd.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, ElasticIsotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold kernel values
  simd_type rho;
  simd_type mu;
  simd_type kappa;
  simd_type rhop;
  simd_type alpha;
  simd_type beta;

  if constexpr (using_simd) {
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = static_cast<type_real>(2.5) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      mu[i] = static_cast<type_real>(3.0) +
              static_cast<type_real>(i) * static_cast<type_real>(0.1);
      kappa[i] = static_cast<type_real>(4.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      rhop[i] = static_cast<type_real>(5.0) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);
      alpha[i] = static_cast<type_real>(6.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      beta[i] = static_cast<type_real>(7.0) +
                static_cast<type_real>(i) * static_cast<type_real>(0.1);
    }
  } else {
    // For scalar case, we need direct assignment
    rho = static_cast<type_real>(2.5);
    mu = static_cast<type_real>(3.0);
    kappa = static_cast<type_real>(4.0);
    rhop = static_cast<type_real>(5.0);
    alpha = static_cast<type_real>(6.0);
    beta = static_cast<type_real>(7.0);
  }

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, using_simd>
      kernels(rho, mu, kappa, rhop, alpha, beta);

  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.rho() - rho) < tol))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.mu() - mu) < tol))
      << ExpectedGot(mu, kernels.mu());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.kappa() - kappa) < tol))
      << ExpectedGot(kappa, kernels.kappa());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.rhop() - rhop) < tol))
      << ExpectedGot(rhop, kernels.rhop());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.alpha() - alpha) < tol))
      << ExpectedGot(alpha, kernels.alpha());
  EXPECT_TRUE(
      specfem::datatype::all_of(Kokkos::abs(kernels.beta() - beta) < tol))
      << ExpectedGot(beta, kernels.beta());
}
