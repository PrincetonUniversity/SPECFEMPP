#include "jacobian/shape_functions.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using specfem::jacobian::define_shape_functions;
using specfem::jacobian::define_shape_functions_derivatives;

TEST(ShapeFunctions, FourNodeSumToOne) {
  auto shape = define_shape_functions(0.0, 0.0, 4);
  type_real sum = 0;
  for (auto v : shape)
    sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6);
  EXPECT_EQ(shape.size(), 4);
}

TEST(ShapeFunctions, NineNodeSumToOne) {
  auto shape = define_shape_functions(0.5, -0.5, 9);
  type_real sum = 0;
  for (auto v : shape)
    sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6);
  EXPECT_EQ(shape.size(), 9);
}

TEST(ShapeFunctions, FourNodeDerivativesSumToZero) {
  auto ders = define_shape_functions_derivatives(0.0, 0.0, 4);
  ASSERT_EQ(ders.size(), 2); // ndim = 2
  ASSERT_EQ(ders[0].size(), 4);
  ASSERT_EQ(ders[1].size(), 4);
  type_real sum_xi = 0, sum_gamma = 0;
  for (int i = 0; i < 4; ++i) {
    sum_xi += ders[0][i];
    sum_gamma += ders[1][i];
  }
  EXPECT_NEAR(sum_xi, 0.0, 1e-6);
  EXPECT_NEAR(sum_gamma, 0.0, 1e-6);
}

TEST(ShapeFunctions, NineNodeDerivativesSumToZero) {
  auto ders = define_shape_functions_derivatives(0.1, -0.2, 9);
  ASSERT_EQ(ders.size(), 2); // ndim = 2
  ASSERT_EQ(ders[0].size(), 9);
  ASSERT_EQ(ders[1].size(), 9);
  type_real sum_xi = 0, sum_gamma = 0;
  for (int i = 0; i < 9; ++i) {
    sum_xi += ders[0][i];
    sum_gamma += ders[1][i];
  }
  EXPECT_NEAR(sum_xi, 0.0, 1e-6);
  EXPECT_NEAR(sum_gamma, 0.0, 1e-6);
}

TEST(ShapeFunctions, InvalidNodeCountThrows) {
  EXPECT_THROW(define_shape_functions(0.0, 0.0, 5), std::invalid_argument);
  EXPECT_THROW(define_shape_functions_derivatives(0.0, 0.0, 7),
               std::invalid_argument);
}

TEST(ShapeFunctions, KnownValues) {
  // Test at corners for 4-node
  auto shape4 = define_shape_functions(-1.0, -1.0, 4);
  EXPECT_NEAR(shape4[0], 1.0, 1e-6);
  for (int i = 1; i < 4; ++i)
    EXPECT_NEAR(shape4[i], 0.0, 1e-6);

  shape4 = define_shape_functions(1.0, -1.0, 4);
  EXPECT_NEAR(shape4[1], 1.0, 1e-6);
  for (int i = 0; i < 4; ++i)
    if (i != 1)
      EXPECT_NEAR(shape4[i], 0.0, 1e-6);

  shape4 = define_shape_functions(1.0, 1.0, 4);
  EXPECT_NEAR(shape4[2], 1.0, 1e-6);
  for (int i = 0; i < 4; ++i)
    if (i != 2)
      EXPECT_NEAR(shape4[i], 0.0, 1e-6);

  shape4 = define_shape_functions(-1.0, 1.0, 4);
  EXPECT_NEAR(shape4[3], 1.0, 1e-6);
  for (int i = 0; i < 4; ++i)
    if (i != 3)
      EXPECT_NEAR(shape4[i], 0.0, 1e-6);

  // Test midpoints for 4-node
  shape4 = define_shape_functions(0.0, 0.0, 4);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(shape4[i], 0.25, 1e-6);

  shape4 = define_shape_functions(0.0, -1.0, 4);
  EXPECT_NEAR(shape4[0], 0.5, 1e-6);
  EXPECT_NEAR(shape4[1], 0.5, 1e-6);
  EXPECT_NEAR(shape4[2], 0.0, 1e-6);
  EXPECT_NEAR(shape4[3], 0.0, 1e-6);

  shape4 = define_shape_functions(1.0, 0.0, 4);
  EXPECT_NEAR(shape4[1], 0.5, 1e-6);
  EXPECT_NEAR(shape4[2], 0.5, 1e-6);
  EXPECT_NEAR(shape4[0], 0.0, 1e-6);
  EXPECT_NEAR(shape4[3], 0.0, 1e-6);

  shape4 = define_shape_functions(0.0, 1.0, 4);
  EXPECT_NEAR(shape4[2], 0.5, 1e-6);
  EXPECT_NEAR(shape4[3], 0.5, 1e-6);
  EXPECT_NEAR(shape4[0], 0.0, 1e-6);
  EXPECT_NEAR(shape4[1], 0.0, 1e-6);

  shape4 = define_shape_functions(-1.0, 0.0, 4);
  EXPECT_NEAR(shape4[3], 0.5, 1e-6);
  EXPECT_NEAR(shape4[0], 0.5, 1e-6);
  EXPECT_NEAR(shape4[1], 0.0, 1e-6);
  EXPECT_NEAR(shape4[2], 0.0, 1e-6);

  auto shape9 = define_shape_functions(-1.0, -1.0, 9);
  EXPECT_NEAR(shape9[0], 1.0, 1e-6);
  for (int i = 1; i < 9; ++i)
    EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(1.0, -1.0, 9);
  EXPECT_NEAR(shape9[1], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 1)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(1.0, 1.0, 9);
  EXPECT_NEAR(shape9[2], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 2)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(-1.0, 1.0, 9);
  EXPECT_NEAR(shape9[3], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 3)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  // Test at midpoints for 9-node
  shape9 = define_shape_functions(0.0, 0.0, 9);
  EXPECT_NEAR(shape9[8], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 8)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(0.0, -1.0, 9);
  EXPECT_NEAR(shape9[4], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 4)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(1.0, 0.0, 9);
  EXPECT_NEAR(shape9[5], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 5)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(0.0, 1.0, 9);
  EXPECT_NEAR(shape9[6], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 6)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);

  shape9 = define_shape_functions(-1.0, 0.0, 9);
  EXPECT_NEAR(shape9[7], 1.0, 1e-6);
  for (int i = 0; i < 9; ++i)
    if (i != 7)
      EXPECT_NEAR(shape9[i], 0.0, 1e-6);
}
