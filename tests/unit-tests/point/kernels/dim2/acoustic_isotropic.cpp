#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
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
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold kernel values
  simd_type rho;
  simd_type kappa;
  simd_type expected_rhop;
  simd_type expected_alpha;

  if constexpr (using_simd) {
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = static_cast<type_real>(2.5) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      kappa[i] = static_cast<type_real>(3.0) +
                 static_cast<type_real>(i) * static_cast<type_real>(0.1);
      expected_rhop[i] =
          static_cast<type_real>(rho[i]) * static_cast<type_real>(kappa[i]);
      expected_alpha[i] =
          static_cast<type_real>(2.0) * static_cast<type_real>(kappa[i]);
    }
  } else {
    // For scalar case, we need direct assignment
    rho = static_cast<type_real>(2.5);
    kappa = static_cast<type_real>(3.0);
    expected_rhop = rho * kappa;
    expected_alpha = static_cast<type_real>(2.0) * kappa;
  }

  // Create the kernels object
  specfem::point::kernels<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, using_simd>
      kernels(rho, kappa);

  EXPECT_TRUE(specfem::utilities::is_close(kernels.rho(), rho))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.kappa(), kappa))
      << ExpectedGot(kappa, kernels.kappa());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.rhop(), expected_rhop))
      << ExpectedGot(expected_rhop, kernels.rhop());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.alpha(), expected_alpha))
      << ExpectedGot(expected_alpha, kernels.alpha());
}
