#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

// ============================================================================
// 3D Elastic Isotropic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, ElasticIsotropic3D) {

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
      rho[i] = static_cast<type_real>(2.7) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      mu[i] = static_cast<type_real>(30.0) +
              static_cast<type_real>(i) * static_cast<type_real>(0.5);
      kappa[i] = static_cast<type_real>(40.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.5);
      rhop[i] = static_cast<type_real>(50.0) +
                static_cast<type_real>(i) * static_cast<type_real>(1.0);
      alpha[i] = static_cast<type_real>(60.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(1.0);
      beta[i] = static_cast<type_real>(70.0) +
                static_cast<type_real>(i) * static_cast<type_real>(1.0);
    }
  } else {
    // For scalar case, we need direct assignment
    rho = static_cast<type_real>(2.7);
    mu = static_cast<type_real>(30.0);
    kappa = static_cast<type_real>(40.0);
    rhop = static_cast<type_real>(50.0);
    alpha = static_cast<type_real>(60.0);
    beta = static_cast<type_real>(70.0);
  }

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic, using_simd>
      kernels(rho, mu, kappa, rhop, alpha, beta);

  EXPECT_TRUE(specfem::utilities::is_close(kernels.rho(), rho))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.mu(), mu))
      << ExpectedGot(mu, kernels.mu());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.kappa(), kappa))
      << ExpectedGot(kappa, kernels.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhop(), rhop))
      << ExpectedGot(rhop, kernels.rhop());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.alpha(), alpha))
      << ExpectedGot(alpha, kernels.alpha());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.beta(), beta))
      << ExpectedGot(beta, kernels.beta());
}
