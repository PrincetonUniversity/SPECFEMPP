#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/point/field_derivatives.hpp"
#include "specfem/setup.hpp"
#include <gtest/gtest.h>

namespace {

TEST(Strain, Elastic3D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic;

  using FieldDerivativesType =
      specfem::point::field_derivatives<dimension, elasticTag, false>;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 3, 3, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0; // ∂u_x/∂x
  field_derivatives.du(1, 1) = 2.0; // ∂u_y/∂y
  field_derivatives.du(2, 2) = 3.0; // ∂u_z/∂z
  field_derivatives.du(0, 1) = 4.0; // ∂u_x/∂y
  field_derivatives.du(1, 0) = 5.0; // ∂u_y/∂x
  field_derivatives.du(0, 2) = 6.0; // ∂u_x/∂z
  field_derivatives.du(2, 0) = 7.0; // ∂u_z/∂x
  field_derivatives.du(1, 2) = 8.0; // ∂u_y/∂z
  field_derivatives.du(2, 1) = 9.0; // ∂u_z/∂y

  const auto strain =
      specfem::medium_physics::compute_strain(field_derivatives);

  StrainType expected;
  expected(0, 0) = 1.0;               // ε_xx
  expected(1, 1) = 2.0;               // ε_yy
  expected(2, 2) = 3.0;               // ε_zz
  expected(0, 1) = 0.5 * (4.0 + 5.0); // ε_xy
  expected(1, 0) = 0.5 * (4.0 + 5.0); // ε_yx
  expected(0, 2) = 0.5 * (6.0 + 7.0); // ε_xz
  expected(2, 0) = 0.5 * (6.0 + 7.0); // ε_zx
  expected(1, 2) = 0.5 * (8.0 + 9.0); // ε_yz
  expected(2, 1) = 0.5 * (8.0 + 9.0); // ε_zy

  EXPECT_TRUE(strain == expected);
}

TEST(Strain, Elastic3D_Zero) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic;

  using FieldDerivativesType =
      specfem::point::field_derivatives<dimension, elasticTag, false>;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 3, 3, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 0.0;
  field_derivatives.du(1, 1) = 0.0;
  field_derivatives.du(2, 2) = 0.0;
  field_derivatives.du(0, 1) = 0.0;
  field_derivatives.du(1, 0) = 0.0;
  field_derivatives.du(0, 2) = 0.0;
  field_derivatives.du(2, 0) = 0.0;
  field_derivatives.du(1, 2) = 0.0;
  field_derivatives.du(2, 1) = 0.0;

  const auto strain =
      specfem::medium_physics::compute_strain(field_derivatives);

  StrainType expected;
  expected(0, 0) = 0.0;
  expected(1, 1) = 0.0;
  expected(2, 2) = 0.0;
  expected(0, 1) = 0.0;
  expected(1, 0) = 0.0;
  expected(0, 2) = 0.0;
  expected(2, 0) = 0.0;
  expected(1, 2) = 0.0;
  expected(2, 1) = 0.0;

  EXPECT_TRUE(strain == expected);
}

TEST(Strain, DeviatoricElastic3D_Basic) {
  static constexpr auto dimension = specfem::element::dimension_tag::dim3;
  static constexpr auto elasticTag = specfem::element::medium_tag::elastic;

  using FieldDerivativesType =
      specfem::point::field_derivatives<dimension, elasticTag, false>;
  using StrainType =
      specfem::datatype::TensorPointViewType<type_real, 3, 3, false>;

  FieldDerivativesType field_derivatives;
  field_derivatives.du(0, 0) = 1.0; // ε_xx = 1
  field_derivatives.du(1, 1) = 2.0; // ε_yy = 2
  field_derivatives.du(2, 2) = 3.0; // ε_zz = 3
  field_derivatives.du(0, 1) = 0.4;
  field_derivatives.du(1, 0) = 0.6;
  field_derivatives.du(0, 2) = 0.2;
  field_derivatives.du(2, 0) = 0.8;
  field_derivatives.du(1, 2) = 0.1;
  field_derivatives.du(2, 1) = 0.9;

  const auto dev_strain =
      specfem::medium_physics::compute_deviatoric_strain(field_derivatives);

  // trace = 1 + 2 + 3 = 6; trace/3 = 2
  const type_real third_trace = 2.0;

  StrainType expected;
  expected(0, 0) = 1.0 - third_trace; // ε_xx^dev
  expected(1, 1) = 2.0 - third_trace; // ε_yy^dev (= 0)
  expected(2, 2) = 3.0 - third_trace; // ε_zz^dev
  expected(0, 1) = 0.5 * (0.4 + 0.6); // ε_xy (= 0.5)
  expected(1, 0) = 0.5 * (0.4 + 0.6);
  expected(0, 2) = 0.5 * (0.2 + 0.8); // ε_xz (= 0.5)
  expected(2, 0) = 0.5 * (0.2 + 0.8);
  expected(1, 2) = 0.5 * (0.1 + 0.9); // ε_yz (= 0.5)
  expected(2, 1) = 0.5 * (0.1 + 0.9);

  EXPECT_TRUE(dev_strain == expected);

  // Deviatoric trace must be zero in 3D
  const type_real dev_trace =
      dev_strain(0, 0) + dev_strain(1, 1) + dev_strain(2, 2);
  EXPECT_NEAR(static_cast<double>(dev_trace), 0.0, 1e-12)
      << "Deviatoric trace should be zero in 3D, got: " << dev_trace;
}

} // namespace
