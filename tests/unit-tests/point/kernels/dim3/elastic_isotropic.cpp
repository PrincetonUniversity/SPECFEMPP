#include "../kernels_tests.hpp"
#include "datatypes/simd.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
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
      rho_arr[i] = static_cast<type_real>(2.7) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      mu_arr[i] = static_cast<type_real>(30.0) +
                  static_cast<type_real>(i) * static_cast<type_real>(0.5);
      kappa_arr[i] = static_cast<type_real>(40.0) +
                     static_cast<type_real>(i) * static_cast<type_real>(0.5);
      rhop_arr[i] = static_cast<type_real>(50.0) +
                    static_cast<type_real>(i) * static_cast<type_real>(1.0);
      alpha_arr[i] = static_cast<type_real>(60.0) +
                     static_cast<type_real>(i) * static_cast<type_real>(1.0);
      beta_arr[i] = static_cast<type_real>(70.0) +
                    static_cast<type_real>(i) * static_cast<type_real>(1.0);
    }
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    mu.copy_from(mu_arr, Kokkos::Experimental::simd_flag_default);
    kappa.copy_from(kappa_arr, Kokkos::Experimental::simd_flag_default);
    rhop.copy_from(rhop_arr, Kokkos::Experimental::simd_flag_default);
    alpha.copy_from(alpha_arr, Kokkos::Experimental::simd_flag_default);
    beta.copy_from(beta_arr, Kokkos::Experimental::simd_flag_default);
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
