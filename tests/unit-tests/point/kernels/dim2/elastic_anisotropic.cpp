#include "../kernels_tests.hpp"
#include "datatypes/simd.hpp"
#include "specfem/point/kernels.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

const type_real tol = 1e-6; ///< Tolerance for floating point comparisons

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
    c11.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c13.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c15.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c33.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c35.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
    c55.copy_from(rho_arr, Kokkos::Experimental::simd_flag_default);
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

  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.rho() - rho) < tol))
      << ExpectedGot(rho, kernels.rho());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c11() - c11) < tol))
      << ExpectedGot(c11, kernels.c11());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c13() - c13) < tol))
      << ExpectedGot(c13, kernels.c13());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c15() - c15) < tol))
      << ExpectedGot(c15, kernels.c15());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c33() - c33) < tol))
      << ExpectedGot(c33, kernels.c33());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c35() - c35) < tol))
      << ExpectedGot(c35, kernels.c35());
  EXPECT_TRUE(specfem::datatype::all_of(Kokkos::abs(kernels.c55() - c55) < tol))
      << ExpectedGot(c55, kernels.c55());
}
