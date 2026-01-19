#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Acoustic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, AcousticIsotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold kernel values
  simd_type rho;
  simd_type kappa;
  simd_type expected_rhop;
  simd_type expected_alpha;

  if constexpr (using_simd) {
    T rho_arr[simd_size];
    T kappa_arr[simd_size];
    T expected_rhop_arr[simd_size];
    T expected_alpha_arr[simd_size];
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] = static_cast<type_real>(2.5) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      kappa_arr[i] = static_cast<type_real>(3.0) +
                     static_cast<type_real>(i) * static_cast<type_real>(0.1);
      expected_rhop_arr[i] = static_cast<type_real>(rho_arr[i]) *
                             static_cast<type_real>(kappa_arr[i]);
      expected_alpha_arr[i] =
          static_cast<type_real>(2.0) * static_cast<type_real>(kappa_arr[i]);
    }
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    kappa.copy_from(kappa_arr, Kokkos::Experimental::simd_flag_default);
    expected_rhop.copy_from(expected_rhop_arr,
                            Kokkos::Experimental::simd_flag_default);
    expected_alpha.copy_from(expected_alpha_arr,
                             Kokkos::Experimental::simd_flag_default);
  } else {
    // For scalar case, we need direct assignment
    rho = static_cast<type_real>(2.5);
    kappa = static_cast<type_real>(3.0);
    expected_rhop = rho * kappa;
    expected_alpha = static_cast<type_real>(2.0) * kappa;
  }

  // Create the kernels object
  using PointKernelType = specfem::point::kernels<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::property_tag::isotropic, using_simd>;
  PointKernelType kernels(rho, kappa);
  PointKernelType kernels2;

  kernels2.rho() = rho;
  kernels2.kappa() = kappa;
  kernels2.rhop() = expected_rhop;
  kernels2.alpha() = expected_alpha;

  simd_type data[] = { rho, kappa, expected_rhop, expected_alpha };

  PointKernelType kernels3(data);
  PointKernelType kernels4(rho);

  EXPECT_TRUE(specfem::utilities::is_close(kernels.rho(), rho))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.kappa(), kappa))
      << ExpectedGot(kappa, kernels.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhop(), expected_rhop))
      << ExpectedGot(expected_rhop, kernels.rhop());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.alpha(), expected_alpha))
      << ExpectedGot(expected_alpha, kernels.alpha());
  EXPECT_TRUE(kernels == kernels2)
      << ExpectedGot(kernels2.rho(), kernels.rho())
      << ExpectedGot(kernels2.kappa(), kernels.kappa());
  EXPECT_TRUE(kernels2 == kernels3)
      << ExpectedGot(kernels3.rho(), kernels2.rho())
      << ExpectedGot(kernels3.kappa(), kernels2.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rho(), rho))
      << ExpectedGot(rho, kernels4.rho());
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.kappa(), rho))
      << ExpectedGot(rho, kernels4.kappa());
}
