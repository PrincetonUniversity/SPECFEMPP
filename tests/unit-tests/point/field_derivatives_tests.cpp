#include "SPECFEM_Environment.hpp"
#include "specfem/enums.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include "specfem/utilities.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace specfem;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

template <bool UseSIMD>
class FieldDerivativesTestUntyped : public ::testing::Test {};

template <typename T>
class FieldDerivativesTest : public FieldDerivativesTestUntyped<T::value> {};

TYPED_TEST_SUITE(FieldDerivativesTest, TestTypes);

// ---------------------------------------------------------------------------
// Helper: Tags aliases
// ---------------------------------------------------------------------------

// dim2, elastic_psv (2 components x 2 dimensions)
template <bool UseSIMD>
using Tags2D =
    specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                        specfem::element::medium_tag::elastic_psv,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none, UseSIMD>;

// dim3, elastic (3 components x 3 dimensions)
template <bool UseSIMD>
using Tags3D =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none, UseSIMD>;

// ---------------------------------------------------------------------------
// operator+ — dim2
// ---------------------------------------------------------------------------

TYPED_TEST(FieldDerivativesTest, Dim2_AdditionZero) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a, b;
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k) {
      a.du(i, k) = datatype{ 0 };
      b.du(i, k) = datatype{ 0 };
    }

  FD c = a + b;
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k)
      EXPECT_TRUE(specfem::utilities::is_close(c.du(i, k), datatype{ 0 }));
}

TYPED_TEST(FieldDerivativesTest, Dim2_AdditionValues) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a, b;
  // a.du = [[1,2],[3,4]], b.du = [[5,6],[7,8]]
  a.du(0, 0) = datatype{ 1 };
  a.du(0, 1) = datatype{ 2 };
  a.du(1, 0) = datatype{ 3 };
  a.du(1, 1) = datatype{ 4 };
  b.du(0, 0) = datatype{ 5 };
  b.du(0, 1) = datatype{ 6 };
  b.du(1, 0) = datatype{ 7 };
  b.du(1, 1) = datatype{ 8 };

  FD c = a + b;
  EXPECT_TRUE(specfem::utilities::is_close(c.du(0, 0), datatype{ 6 }));
  EXPECT_TRUE(specfem::utilities::is_close(c.du(0, 1), datatype{ 8 }));
  EXPECT_TRUE(specfem::utilities::is_close(c.du(1, 0), datatype{ 10 }));
  EXPECT_TRUE(specfem::utilities::is_close(c.du(1, 1), datatype{ 12 }));
}

// ---------------------------------------------------------------------------
// operator* (right scalar) — dim2
// ---------------------------------------------------------------------------

TYPED_TEST(FieldDerivativesTest, Dim2_ScalarMultiplyRight) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a;
  a.du(0, 0) = datatype{ 2 };
  a.du(0, 1) = datatype{ 3 };
  a.du(1, 0) = datatype{ 4 };
  a.du(1, 1) = datatype{ 5 };

  FD b = a * static_cast<type_real>(2);
  EXPECT_TRUE(specfem::utilities::is_close(b.du(0, 0), datatype{ 4 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(0, 1), datatype{ 6 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(1, 0), datatype{ 8 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(1, 1), datatype{ 10 }));
}

TYPED_TEST(FieldDerivativesTest, Dim2_ScalarMultiplyLeft) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a;
  a.du(0, 0) = datatype{ 2 };
  a.du(0, 1) = datatype{ 3 };
  a.du(1, 0) = datatype{ 4 };
  a.du(1, 1) = datatype{ 5 };

  FD b = static_cast<type_real>(3) * a;
  EXPECT_TRUE(specfem::utilities::is_close(b.du(0, 0), datatype{ 6 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(0, 1), datatype{ 9 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(1, 0), datatype{ 12 }));
  EXPECT_TRUE(specfem::utilities::is_close(b.du(1, 1), datatype{ 15 }));
}

TYPED_TEST(FieldDerivativesTest, Dim2_ScalarMultiplyZero) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a;
  a.du(0, 0) = datatype{ 7 };
  a.du(0, 1) = datatype{ 8 };
  a.du(1, 0) = datatype{ 9 };
  a.du(1, 1) = datatype{ 10 };

  FD b = a * static_cast<type_real>(0);
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k)
      EXPECT_TRUE(specfem::utilities::is_close(b.du(i, k), datatype{ 0 }));
}

// ---------------------------------------------------------------------------
// du + scalar * dv (Taylor step) — dim2
// ---------------------------------------------------------------------------

TYPED_TEST(FieldDerivativesTest, Dim2_TaylorStep) {
  // Mirrors attenuation usage: du_att = du + deltat * dv
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags2D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD du, dv;
  du.du(0, 0) = datatype{ 1 };
  du.du(0, 1) = datatype{ 2 };
  du.du(1, 0) = datatype{ 3 };
  du.du(1, 1) = datatype{ 4 };
  dv.du(0, 0) = datatype{ 10 };
  dv.du(0, 1) = datatype{ 20 };
  dv.du(1, 0) = datatype{ 30 };
  dv.du(1, 1) = datatype{ 40 };

  const type_real deltat = static_cast<type_real>(0.5);
  FD du_att = du + deltat * dv;

  // Expected: du + 0.5 * dv
  EXPECT_TRUE(
      specfem::utilities::is_close(du_att.du(0, 0), datatype{ 6 })); // 1+5
  EXPECT_TRUE(
      specfem::utilities::is_close(du_att.du(0, 1), datatype{ 12 })); // 2+10
  EXPECT_TRUE(
      specfem::utilities::is_close(du_att.du(1, 0), datatype{ 18 })); // 3+15
  EXPECT_TRUE(
      specfem::utilities::is_close(du_att.du(1, 1), datatype{ 24 })); // 4+20
}

// ---------------------------------------------------------------------------
// operator+ — dim3
// ---------------------------------------------------------------------------

TYPED_TEST(FieldDerivativesTest, Dim3_AdditionValues) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags3D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD a, b;
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k) {
      a.du(i, k) = datatype{ static_cast<type_real>(i * 3 + k + 1) };
      b.du(i, k) = datatype{ static_cast<type_real>(1) };
    }

  FD c = a + b;
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k) {
      datatype expected{ static_cast<type_real>(i * 3 + k + 2) };
      EXPECT_TRUE(specfem::utilities::is_close(c.du(i, k), expected))
          << ExpectedGot(static_cast<type_real>(i * 3 + k + 2), c.du(i, k));
    }
}

// ---------------------------------------------------------------------------
// Taylor step — dim3
// ---------------------------------------------------------------------------

TYPED_TEST(FieldDerivativesTest, Dim3_TaylorStep) {
  constexpr bool using_simd = TypeParam::value;
  using FD = specfem::point::field_derivatives<Tags3D<using_simd> >;
  using datatype =
      typename specfem::datatype::simd<type_real, using_simd>::datatype;

  FD du, dv;
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k) {
      du.du(i, k) = datatype{ 1 };
      dv.du(i, k) = datatype{ 2 };
    }

  const type_real deltat = static_cast<type_real>(1);
  FD du_att = du + deltat * dv;

  // Expected: 1 + 1*2 = 3 for every entry
  for (int i = 0; i < FD::components; ++i)
    for (int k = 0; k < FD::num_dimensions; ++k)
      EXPECT_TRUE(specfem::utilities::is_close(du_att.du(i, k), datatype{ 3 }))
          << ExpectedGot(3.0, du_att.du(i, k));
}
