#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ============================================================================
// 2D Elastic Anisotropic Tests
// ============================================================================
TYPED_TEST(PointKernelsTest, ElasticAnisotropic2D) {

  constexpr bool using_simd = TypeParam::value; ///< Use SIMD if true

  // Get the SIMD size from the implementation
  using simd_type =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Variables to hold kernel values
  simd_type rho;
  simd_type c11;
  simd_type c13;
  simd_type c15;
  simd_type c33;
  simd_type c35;
  simd_type c55;

  if constexpr (using_simd) {
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho[i] = static_cast<type_real>(2.5) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c11[i] = static_cast<type_real>(10.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c13[i] = static_cast<type_real>(11.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c15[i] = static_cast<type_real>(12.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c33[i] = static_cast<type_real>(13.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c35[i] = static_cast<type_real>(14.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c55[i] = static_cast<type_real>(15.0) +
               static_cast<type_real>(i) * static_cast<type_real>(0.1);
    }
  } else {
    // For scalar case, we need direct assignment
    rho = static_cast<type_real>(2.5);
    c11 = static_cast<type_real>(10.0);
    c13 = static_cast<type_real>(11.0);
    c15 = static_cast<type_real>(12.0);
    c33 = static_cast<type_real>(13.0);
    c35 = static_cast<type_real>(14.0);
    c55 = static_cast<type_real>(15.0);
  }

  // Create the kernels object
  specfem::point::kernels<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, using_simd>
      kernels(rho, c11, c13, c15, c33, c35, c55);

  EXPECT_TRUE(specfem::utilities::is_close(kernels.rho(), rho))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c11(), c11))
      << ExpectedGot(c11, kernels.c11());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c13(), c13))
      << ExpectedGot(c13, kernels.c13());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c15(), c15))
      << ExpectedGot(c15, kernels.c15());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c33(), c33))
      << ExpectedGot(c33, kernels.c33());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c35(), c35))
      << ExpectedGot(c35, kernels.c35());
  EXPECT_TRUE(specfem::utilities::is_close(kernels.c55(), c55))
      << ExpectedGot(c55, kernels.c55());
}
