// Tests for specfem::point::strain

#include "specfem/element.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/point/strain.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

template <bool UseSIMD> class PointStrainTestUntyped : public ::testing::Test {
protected:
  void SetUp() override {
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};

template <typename T>
class PointStrainTest : public PointStrainTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointStrainTest, TestTypes);

// ============================================================
// Static property checks
// ============================================================

TYPED_TEST(PointStrainTest, StaticProperties2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  EXPECT_EQ(s_type::dimension_tag, element::dimension_tag::dim2);
  EXPECT_EQ(s_type::medium_tag, element::medium_tag::elastic_psv);
  EXPECT_EQ(s_type::using_simd, using_simd);
}

TYPED_TEST(PointStrainTest, StaticProperties3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  EXPECT_EQ(s_type::dimension_tag, element::dimension_tag::dim3);
  EXPECT_EQ(s_type::medium_tag, element::medium_tag::elastic);
  EXPECT_EQ(s_type::using_simd, using_simd);
}

// ============================================================
// Default constructor — all fields zero
// ============================================================

TYPED_TEST(PointStrainTest, DefaultConstructor2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  s_type s;

  if constexpr (!using_simd) {
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0)));
  }
}

TYPED_TEST(PointStrainTest, DefaultConstructor3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  s_type s;

  if constexpr (!using_simd) {
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yy, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xy, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yz, type_real(0)));
  }
}

// ============================================================
// Component value constructor
// ============================================================

TYPED_TEST(PointStrainTest, ValueConstructor2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    typename s_type::scalar_type exx(1.0), ezz(2.0), exz(0.5);
    s_type s(exx, ezz, exz);

    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, exx))
        << ExpectedGot(exx, s.epsilon_xx);
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, ezz))
        << ExpectedGot(ezz, s.epsilon_zz);
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, exz))
        << ExpectedGot(exz, s.epsilon_xz);
  }
}

TYPED_TEST(PointStrainTest, ValueConstructor3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    typename s_type::scalar_type exx(1.0), eyy(2.0), ezz(3.0), exy(0.4),
        exz(0.5), eyz(0.6);
    s_type s(exx, eyy, ezz, exy, exz, eyz);

    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, exx));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yy, eyy));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, ezz));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xy, exy));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, exz));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yz, eyz));
  }
}

// ============================================================
// field_derivatives constructor — verifies strain computation
// ============================================================

TYPED_TEST(PointStrainTest, FieldDerivativesConstructor2D) {
  constexpr bool using_simd = TypeParam::value;

  using fd_type =
      point::field_derivatives<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;
  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    // du(k, i) = ∂u_k/∂x_i
    typename fd_type::value_type du(0.0);
    du(0, 0) = type_real(2.0); // ∂ux/∂x
    du(0, 1) = type_real(0.6); // ∂ux/∂z
    du(1, 0) = type_real(0.4); // ∂uz/∂x
    du(1, 1) = type_real(3.0); // ∂uz/∂z

    s_type s((fd_type(du)));

    // epsilon_xx = du(0,0) = 2.0
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(2.0)))
        << ExpectedGot(type_real(2.0), s.epsilon_xx);
    // epsilon_zz = du(1,1) = 3.0
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(3.0)))
        << ExpectedGot(type_real(3.0), s.epsilon_zz);
    // epsilon_xz = (du(0,1) + du(1,0)) / 2 = (0.6 + 0.4) / 2 = 0.5
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0.5)))
        << ExpectedGot(type_real(0.5), s.epsilon_xz);
  }
}

TYPED_TEST(PointStrainTest, FieldDerivativesConstructor3D) {
  constexpr bool using_simd = TypeParam::value;

  using fd_type =
      point::field_derivatives<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;
  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    typename fd_type::value_type du(0.0);
    du(0, 0) = type_real(1.0); // ∂ux/∂x
    du(1, 1) = type_real(2.0); // ∂uy/∂y
    du(2, 2) = type_real(3.0); // ∂uz/∂z
    du(0, 1) = type_real(0.4); // ∂ux/∂y
    du(1, 0) = type_real(0.6); // ∂uy/∂x
    du(0, 2) = type_real(0.2); // ∂ux/∂z
    du(2, 0) = type_real(0.8); // ∂uz/∂x
    du(1, 2) = type_real(0.1); // ∂uy/∂z
    du(2, 1) = type_real(0.9); // ∂uz/∂y

    s_type s((fd_type(du)));

    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(1.0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yy, type_real(2.0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(3.0)));
    // epsilon_xy = (0.4 + 0.6) / 2 = 0.5
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xy, type_real(0.5)));
    // epsilon_xz = (0.2 + 0.8) / 2 = 0.5
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0.5)));
    // epsilon_yz = (0.1 + 0.9) / 2 = 0.5
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yz, type_real(0.5)));
  }
}

// ============================================================
// trace() method
// ============================================================

TYPED_TEST(PointStrainTest, Trace2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    typename s_type::scalar_type exx(3.0), ezz(5.0), exz(1.0);
    s_type s(exx, ezz, exz);

    // trace = exx + ezz = 8.0
    EXPECT_TRUE(specfem::utilities::is_close(s.trace(), type_real(8.0)))
        << ExpectedGot(type_real(8.0), s.trace());
  }
}

TYPED_TEST(PointStrainTest, Trace3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    typename s_type::scalar_type exx(1.0), eyy(2.0), ezz(3.0), exy(0.4),
        exz(0.5), eyz(0.6);
    s_type s(exx, eyy, ezz, exy, exz, eyz);

    // trace = 1 + 2 + 3 = 6.0
    EXPECT_TRUE(specfem::utilities::is_close(s.trace(), type_real(6.0)))
        << ExpectedGot(type_real(6.0), s.trace());
  }
}

// ============================================================
// deviatoric() method
// ============================================================

TYPED_TEST(PointStrainTest, Deviatoric2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    // trace = 3 + 5 = 8, so trace/3 = 8/3
    typename s_type::scalar_type exx(3.0), ezz(5.0), exz(1.0);
    s_type s(exx, ezz, exz);
    s_type dev = s.deviatoric();

    const type_real third_trace = type_real(8.0) / type_real(3.0);
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_xx, exx - third_trace))
        << ExpectedGot(exx - third_trace, dev.epsilon_xx);
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_zz, ezz - third_trace))
        << ExpectedGot(ezz - third_trace, dev.epsilon_zz);
    // shear unchanged
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_xz, exz))
        << ExpectedGot(exz, dev.epsilon_xz);
    // In 2D plane-strain the stored deviatoric trace equals trace/3,
    // not zero — the missing ε_yy^dev = -trace/3 makes the full 3D sum vanish.
    const type_real expected_dev_trace = s.trace() / type_real(3.0);
    EXPECT_TRUE(specfem::utilities::is_close(dev.trace(), expected_dev_trace))
        << "deviatoric trace: " << ExpectedGot(expected_dev_trace, dev.trace());
  }
}

TYPED_TEST(PointStrainTest, Deviatoric3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    // trace = 1 + 2 + 3 = 6, so trace/3 = 2
    typename s_type::scalar_type exx(1.0), eyy(2.0), ezz(3.0), exy(0.4),
        exz(0.5), eyz(0.6);
    s_type s(exx, eyy, ezz, exy, exz, eyz);
    s_type dev = s.deviatoric();

    const type_real third_trace = type_real(2.0); // 6/3
    EXPECT_TRUE(
        specfem::utilities::is_close(dev.epsilon_xx, exx - third_trace));
    EXPECT_TRUE(
        specfem::utilities::is_close(dev.epsilon_yy, eyy - third_trace));
    EXPECT_TRUE(
        specfem::utilities::is_close(dev.epsilon_zz, ezz - third_trace));
    // shear components unchanged
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_xy, exy));
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_xz, exz));
    EXPECT_TRUE(specfem::utilities::is_close(dev.epsilon_yz, eyz));
    // deviatoric trace should be zero
    EXPECT_TRUE(specfem::utilities::is_close(dev.trace(), type_real(0)))
        << "deviatoric trace: " << ExpectedGot(type_real(0), dev.trace());
  }
}

// ============================================================
// init() method — reset to zero
// ============================================================

TYPED_TEST(PointStrainTest, InitMethod2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    s_type s(type_real(5.0), type_real(5.0), type_real(5.0));
    s.init();

    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0)));
  }
}

TYPED_TEST(PointStrainTest, InitMethod3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    s_type s(type_real(5.0), type_real(5.0), type_real(5.0), type_real(5.0),
             type_real(5.0), type_real(5.0));
    s.init();

    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xx, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yy, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_zz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xy, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_xz, type_real(0)));
    EXPECT_TRUE(specfem::utilities::is_close(s.epsilon_yz, type_real(0)));
  }
}

// ============================================================
// Equality operator
// ============================================================

TYPED_TEST(PointStrainTest, EqualityOperator2D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim2,
                               element::medium_tag::elastic_psv, using_simd>;

  if constexpr (!using_simd) {
    s_type s1(type_real(1.0), type_real(2.0), type_real(0.5));
    s_type s2(type_real(1.0), type_real(2.0), type_real(0.5));
    s_type s3(type_real(9.0), type_real(2.0), type_real(0.5));

    EXPECT_TRUE(s1 == s2);
    EXPECT_FALSE(s1 == s3);
  }
}

TYPED_TEST(PointStrainTest, EqualityOperator3D) {
  constexpr bool using_simd = TypeParam::value;

  using s_type = point::strain<element::dimension_tag::dim3,
                               element::medium_tag::elastic, using_simd>;

  if constexpr (!using_simd) {
    s_type s1(type_real(1.0), type_real(2.0), type_real(3.0), type_real(0.4),
              type_real(0.5), type_real(0.6));
    s_type s2(type_real(1.0), type_real(2.0), type_real(3.0), type_real(0.4),
              type_real(0.5), type_real(0.6));
    s_type s3(type_real(9.0), type_real(2.0), type_real(3.0), type_real(0.4),
              type_real(0.5), type_real(0.6));

    EXPECT_TRUE(s1 == s2);
    EXPECT_FALSE(s1 == s3);
  }
}

// ============================================================
// SIMD type verification
// ============================================================

TYPED_TEST(PointStrainTest, SIMDTypeVerification) {
  constexpr bool using_simd = TypeParam::value;

  using s2d = point::strain<element::dimension_tag::dim2,
                            element::medium_tag::elastic_psv, using_simd>;
  using s3d = point::strain<element::dimension_tag::dim3,
                            element::medium_tag::elastic, using_simd>;

  bool simd_match_2d =
      std::is_same_v<typename s2d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_2d);

  bool simd_match_3d =
      std::is_same_v<typename s3d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_3d);
}
