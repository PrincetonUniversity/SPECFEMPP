#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Isotropic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, ElasticIsotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
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
    T rho_arr[simd_size];
    T mu_arr[simd_size];
    T kappa_arr[simd_size];
    T rhop_arr[simd_size];
    T alpha_arr[simd_size];
    T beta_arr[simd_size];
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] = static_cast<type_real>(2.5) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      mu_arr[i] = static_cast<type_real>(3.0) +
                  static_cast<type_real>(i) * static_cast<type_real>(0.1);
      kappa_arr[i] = static_cast<type_real>(4.0) +
                     static_cast<type_real>(i) * static_cast<type_real>(0.1);
      rhop_arr[i] = static_cast<type_real>(5.0) +
                    static_cast<type_real>(i) * static_cast<type_real>(0.1);
      alpha_arr[i] = static_cast<type_real>(6.0) +
                     static_cast<type_real>(i) * static_cast<type_real>(0.1);
      beta_arr[i] = static_cast<type_real>(7.0) +
                    static_cast<type_real>(i) * static_cast<type_real>(0.1);
    }
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    mu.copy_from(mu_arr, Kokkos::Experimental::simd_flag_default);
    kappa.copy_from(kappa_arr, Kokkos::Experimental::simd_flag_default);
    rhop.copy_from(rhop_arr, Kokkos::Experimental::simd_flag_default);
    alpha.copy_from(alpha_arr, Kokkos::Experimental::simd_flag_default);
    beta.copy_from(beta_arr, Kokkos::Experimental::simd_flag_default);
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
  using PointKernelType = specfem::point::kernels<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointKernelType kernels(rho, mu, kappa, rhop, alpha, beta);

  // Additional constructors and assignment tests
  PointKernelType kernels2;
  kernels2.rho() = rho;
  kernels2.mu() = mu;
  kernels2.kappa() = kappa;
  kernels2.rhop() = rhop;
  kernels2.alpha() = alpha;
  kernels2.beta() = beta;

  simd_type data[] = { rho, mu, kappa, rhop, alpha, beta };
  PointKernelType kernels3(data);

  PointKernelType kernels4(rho);

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

  EXPECT_TRUE(kernels == kernels2)
      << ExpectedGot(kernels2.rho(), kernels.rho())
      << ExpectedGot(kernels2.mu(), kernels.mu())
      << ExpectedGot(kernels2.kappa(), kernels.kappa())
      << ExpectedGot(kernels2.rhop(), kernels.rhop())
      << ExpectedGot(kernels2.alpha(), kernels.alpha())
      << ExpectedGot(kernels2.beta(), kernels.beta());
  EXPECT_TRUE(kernels2 == kernels3)
      << ExpectedGot(kernels3.rho(), kernels2.rho())
      << ExpectedGot(kernels3.mu(), kernels2.mu())
      << ExpectedGot(kernels3.kappa(), kernels2.kappa())
      << ExpectedGot(kernels3.rhop(), kernels2.rhop())
      << ExpectedGot(kernels3.alpha(), kernels2.alpha())
      << ExpectedGot(kernels3.beta(), kernels2.beta());

  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rho(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.mu(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.kappa(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rhop(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.alpha(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.beta(), rho));
}
