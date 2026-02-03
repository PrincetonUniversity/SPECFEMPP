#include "../kernels_tests.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
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
  using T = typename specfem::datatype::simd<type_real, using_simd>::base_type;
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
    T rho_arr[simd_size];
    T c11_arr[simd_size];
    T c13_arr[simd_size];
    T c15_arr[simd_size];
    T c33_arr[simd_size];
    T c35_arr[simd_size];
    T c55_arr[simd_size];
    // For SIMD case, we can use array indexing syntax
    for (int i = 0; i < simd_size; ++i) {
      rho_arr[i] = static_cast<type_real>(2.5) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c11_arr[i] = static_cast<type_real>(10.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c13_arr[i] = static_cast<type_real>(11.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c15_arr[i] = static_cast<type_real>(12.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c33_arr[i] = static_cast<type_real>(13.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c35_arr[i] = static_cast<type_real>(14.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
      c55_arr[i] = static_cast<type_real>(15.0) +
                   static_cast<type_real>(i) * static_cast<type_real>(0.1);
    }
    rho.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c11.copy_from(c11_arr, Kokkos::Experimental::simd_flag_default);
    c13.copy_from(c13_arr, Kokkos::Experimental::simd_flag_default);
    c15.copy_from(c15_arr, Kokkos::Experimental::simd_flag_default);
    c33.copy_from(c33_arr, Kokkos::Experimental::simd_flag_default);
    c35.copy_from(c35_arr, Kokkos::Experimental::simd_flag_default);
    c55.copy_from(c55_arr, Kokkos::Experimental::simd_flag_default);
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
  using PointKernelType = specfem::point::kernels<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::property_tag::anisotropic, using_simd>;
  PointKernelType kernels(rho, c11, c13, c15, c33, c35, c55);

  // Additional constructors and assignment tests
  simd_type data[] = { rho, c11, c13, c15, c33, c35, c55 };

  PointKernelType kernels2;
  kernels2.rho() = rho;
  kernels2.c11() = c11;
  kernels2.c13() = c13;
  kernels2.c15() = c15;
  kernels2.c33() = c33;
  kernels2.c35() = c35;
  kernels2.c55() = c55;

  PointKernelType kernels3(data);

  PointKernelType kernels4(rho);

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

  EXPECT_TRUE(kernels == kernels2)
      << ExpectedGot(kernels2.rho(), kernels.rho())
      << ExpectedGot(kernels2.c11(), kernels.c11())
      << ExpectedGot(kernels2.c13(), kernels.c13())
      << ExpectedGot(kernels2.c15(), kernels.c15())
      << ExpectedGot(kernels2.c33(), kernels.c33())
      << ExpectedGot(kernels2.c35(), kernels.c35())
      << ExpectedGot(kernels2.c55(), kernels.c55());
  EXPECT_TRUE(kernels2 == kernels3)
      << ExpectedGot(kernels3.rho(), kernels2.rho())
      << ExpectedGot(kernels3.c11(), kernels2.c11())
      << ExpectedGot(kernels3.c13(), kernels2.c13())
      << ExpectedGot(kernels3.c15(), kernels2.c15())
      << ExpectedGot(kernels3.c33(), kernels2.c33())
      << ExpectedGot(kernels3.c35(), kernels2.c35())
      << ExpectedGot(kernels3.c55(), kernels2.c55());

  EXPECT_TRUE(specfem::utilities::is_close(kernels4.rho(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c11(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c13(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c15(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c33(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c35(), rho));
  EXPECT_TRUE(specfem::utilities::is_close(kernels4.c55(), rho));
}
