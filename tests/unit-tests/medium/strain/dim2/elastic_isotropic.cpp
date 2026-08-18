#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <gtest/gtest.h>

namespace {

TEST(Strain, ElasticPSV2D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::tags::Tags<dimension, PSVTag, false> >;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 2, 2, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0; // ∂u_x/∂x
  field_derivatives.du(1, 1) = 2.0; // ∂u_z/∂z
  field_derivatives.du(0, 1) = 3.0; // ∂u_x/∂z
  field_derivatives.du(1, 0) = 4.0; // ∂u_z/∂x

  const auto strain =
      specfem::medium_physics::compute_strain(field_derivatives);

  StrainType expected;
  expected(0, 0) = 1.0;               // ε_xx
  expected(1, 1) = 2.0;               // ε_zz
  expected(0, 1) = 0.5 * (3.0 + 4.0); // ε_xz (symmetric)
  expected(1, 0) = 0.5 * (3.0 + 4.0); // ε_zx (symmetric)

  EXPECT_TRUE(strain == expected);
}

TEST(Strain, ElasticPSV2D_Zero) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::tags::Tags<dimension, PSVTag, false> >;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 2, 2, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(1, 1) = 0.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;

  const auto strain =
      specfem::medium_physics::compute_strain(field_derivatives);

  StrainType expected;
  expected(0, 0) = 0.0;
  expected(1, 1) = 0.0;
  expected(0, 1) = 0.0;
  expected(1, 0) = 0.0;

  EXPECT_TRUE(strain == expected);
}

TEST(Strain, ElasticSH2D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto SHTag = specfem::element::medium_tag::elastic_sh;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::tags::Tags<dimension, SHTag, false> >;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 1, 2, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 2.0; // ∂u_y/∂x
  field_derivatives.du(0, 1) = 5.0; // ∂u_y/∂z

  const auto strain =
      specfem::medium_physics::compute_strain(field_derivatives);

  StrainType expected;
  expected(0, 0) = 2.0;
  expected(0, 1) = 5.0;

  EXPECT_TRUE(strain == expected);
}

TEST(Strain, DeviatoricElasticPSV2D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim2;
  static constexpr auto PSVTag = specfem::element::medium_tag::elastic_psv;

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::tags::Tags<dimension, PSVTag, false> >;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 2, 2, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 3.0; // → ε_xx = 3
  field_derivatives.du(1, 1) = 5.0; // → ε_zz = 5
  field_derivatives.du(0, 1) = 2.0; // → ε_xz = (2+0)/2 = 1
  field_derivatives.du(1, 0) = 0.0;

  const auto dev_strain =
      specfem::medium_physics::compute_deviatoric_strain(field_derivatives);

  // trace = 3 + 5 = 8; trace/3 = 8/3
  const type_real third_trace = type_real(8.0) / type_real(3.0);
  const type_real eps_xz = 1.0; // (2+0)/2

  StrainType expected;
  expected(0, 0) = 3.0 - third_trace;
  expected(1, 1) = 5.0 - third_trace;
  expected(0, 1) = eps_xz;
  expected(1, 0) = eps_xz;

  EXPECT_TRUE(dev_strain == expected);
}

} // namespace
