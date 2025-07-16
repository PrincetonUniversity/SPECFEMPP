#include "specfem/shape_functions.hpp"
#include "specfem_setup.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using specfem::shape_function::shape_function;
using specfem::shape_function::shape_function_derivatives;

TEST(ShapeFunctions3D, EightNodeSumToOne) {
  auto shape = shape_function(0.0, 0.0, 0.0, 8);
  type_real sum = 0;
  for (auto v : shape)
    sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6);
  EXPECT_EQ(shape.size(), 8);
}

TEST(ShapeFunctions3D, EightNodeDerivativesSumToZero) {
  auto ders = shape_function_derivatives(0.0, 0.0, 0.0, 8);
  ASSERT_EQ(ders.size(), 3); // ndim = 3
  ASSERT_EQ(ders[0].size(), 8);
  ASSERT_EQ(ders[1].size(), 8);
  ASSERT_EQ(ders[2].size(), 8);
  type_real sum_xi = 0, sum_eta = 0, sum_zeta = 0;
  for (int i = 0; i < 8; ++i) {
    sum_xi += ders[0][i];
    sum_eta += ders[1][i];
    sum_zeta += ders[2][i];
  }
  EXPECT_NEAR(sum_xi, 0.0, 1e-6);
  EXPECT_NEAR(sum_eta, 0.0, 1e-6);
  EXPECT_NEAR(sum_zeta, 0.0, 1e-6);
}

TEST(ShapeFunctions3D, InvalidNodeCountThrows) {
  EXPECT_THROW(shape_function(0.0, 0.0, 0.0, 7), std::invalid_argument);
  EXPECT_THROW(shape_function_derivatives(0.0, 0.0, 0.0, 12),
               std::invalid_argument);
}

TEST(ShapeFunctions3D, EightNodeKnownValues) {
  // Test at corners for 8-node
  auto shape8 = shape_function(-1.0, -1.0, -1.0, 8);
  EXPECT_NEAR(shape8[0], 1.0, 1e-6);
  for (int i = 1; i < 8; ++i)
    EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(1.0, -1.0, -1.0, 8);
  EXPECT_NEAR(shape8[1], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 1)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(1.0, 1.0, -1.0, 8);
  EXPECT_NEAR(shape8[2], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 2)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(-1.0, 1.0, -1.0, 8);
  EXPECT_NEAR(shape8[3], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 3)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(-1.0, -1.0, 1.0, 8);
  EXPECT_NEAR(shape8[4], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 4)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(1.0, -1.0, 1.0, 8);
  EXPECT_NEAR(shape8[5], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 5)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(1.0, 1.0, 1.0, 8);
  EXPECT_NEAR(shape8[6], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 6)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  shape8 = shape_function(-1.0, 1.0, 1.0, 8);
  EXPECT_NEAR(shape8[7], 1.0, 1e-6);
  for (int i = 0; i < 8; ++i)
    if (i != 7)
      EXPECT_NEAR(shape8[i], 0.0, 1e-6);

  // Test at center for 8-node
  shape8 = shape_function(0.0, 0.0, 0.0, 8);
  for (int i = 0; i < 8; ++i)
    EXPECT_NEAR(shape8[i], 0.125, 1e-6);
}

TEST(ShapeFunctions3D, TwentySevenNodeSumToOne) {
  auto shape = shape_function(0.0, 0.0, 0.0, 27);
  type_real sum = 0;
  for (auto v : shape)
    sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-6);
  EXPECT_EQ(shape.size(), 27);
}

TEST(ShapeFunctions3D, TwentySevenNodeKnownValues) {
  // Test at corners for 27-node
  auto shape27 = shape_function(-1.0, -1.0, -1.0, 27);
  EXPECT_NEAR(shape27[0], 1.0, 1e-6);
  for (int i = 1; i < 27; ++i)
    EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(1.0, -1.0, -1.0, 27);
  EXPECT_NEAR(shape27[1], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 1)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(1.0, 1.0, -1.0, 27);
  EXPECT_NEAR(shape27[2], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 2)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(-1.0, 1.0, -1.0, 27);
  EXPECT_NEAR(shape27[3], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 3)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(-1.0, -1.0, 1.0, 27);
  EXPECT_NEAR(shape27[4], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 4)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(1.0, -1.0, 1.0, 27);
  EXPECT_NEAR(shape27[5], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 5)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(1.0, 1.0, 1.0, 27);
  EXPECT_NEAR(shape27[6], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 6)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  shape27 = shape_function(-1.0, 1.0, 1.0, 27);
  EXPECT_NEAR(shape27[7], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 7)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Test mid-edge nodes for 27-node
  // Edge between node 0 (-1,-1,-1) and node 1 (1,-1,-1): midpoint (0,-1,-1),
  // index 8
  shape27 = shape_function(0.0, -1.0, -1.0, 27);
  EXPECT_NEAR(shape27[8], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 8)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 1 (1,-1,-1) and node 2 (1,1,-1): midpoint (1,0,-1), index
  // 9
  shape27 = shape_function(1.0, 0.0, -1.0, 27);
  EXPECT_NEAR(shape27[9], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 9)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 2 (1,1,-1) and node 3 (-1,1,-1): midpoint (0,1,-1), index
  // 10
  shape27 = shape_function(0.0, 1.0, -1.0, 27);
  EXPECT_NEAR(shape27[10], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 10)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 3 (-1,1,-1) and node 0 (-1,-1,-1): midpoint (-1,0,-1),
  // index 11
  shape27 = shape_function(-1.0, 0.0, -1.0, 27);
  EXPECT_NEAR(shape27[11], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 11)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 0 (-1,-1,-1) and node 4 (-1,-1,1): midpoint (-1,-1,0),
  // index 12
  shape27 = shape_function(-1.0, -1.0, 0.0, 27);
  EXPECT_NEAR(shape27[12], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 12)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 1 (1,-1,-1) and node 5 (1,-1,1): midpoint (1,-1,0), index
  // 13
  shape27 = shape_function(1.0, -1.0, 0.0, 27);
  EXPECT_NEAR(shape27[13], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 13)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 2 (1,1,-1) and node 6 (1,1,1): midpoint (1,1,0), index 14
  shape27 = shape_function(1.0, 1.0, 0.0, 27);
  EXPECT_NEAR(shape27[14], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 14)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 3 (-1,1,-1) and node 7 (-1,1,1): midpoint (-1,1,0), index
  // 15
  shape27 = shape_function(-1.0, 1.0, 0.0, 27);
  EXPECT_NEAR(shape27[15], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 15)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 4 (-1,-1,1) and node 5 (1,-1,1): midpoint (0,-1,1), index
  // 16
  shape27 = shape_function(0.0, -1.0, 1.0, 27);
  EXPECT_NEAR(shape27[16], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 16)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 5 (1,-1,1) and node 6 (1,1,1): midpoint (1,0,1), index 17
  shape27 = shape_function(1.0, 0.0, 1.0, 27);
  EXPECT_NEAR(shape27[17], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 17)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 6 (1,1,1) and node 7 (-1,1,1): midpoint (0,1,1), index 18
  shape27 = shape_function(0.0, 1.0, 1.0, 27);
  EXPECT_NEAR(shape27[18], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 18)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Edge between node 7 (-1,1,1) and node 4 (-1,-1,1): midpoint (-1,0,1), index
  // 19
  shape27 = shape_function(-1.0, 0.0, 1.0, 27);
  EXPECT_NEAR(shape27[19], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 19)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Test at face center nodes for 27-node
  // Node 20: face center at (0,0,-1)
  shape27 = shape_function(0.0, 0.0, -1.0, 27);
  EXPECT_NEAR(shape27[20], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 20)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 21: face center at (0,-1,0)
  shape27 = shape_function(0.0, -1.0, 0.0, 27);
  EXPECT_NEAR(shape27[21], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 21)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 22: face center at (1,0,0)
  shape27 = shape_function(1.0, 0.0, 0.0, 27);
  EXPECT_NEAR(shape27[22], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 22)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 23: face center at (0,1,0)
  shape27 = shape_function(0.0, 1.0, 0.0, 27);
  EXPECT_NEAR(shape27[23], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 23)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 24: face center at (-1,0,0)
  shape27 = shape_function(-1.0, 0.0, 0.0, 27);
  EXPECT_NEAR(shape27[24], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 24)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 25: face center at (0,0,1)
  shape27 = shape_function(0.0, 0.0, 1.0, 27);
  EXPECT_NEAR(shape27[25], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 25)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);

  // Node 26: center node at (0,0,0)
  shape27 = shape_function(0.0, 0.0, 0.0, 27);
  EXPECT_NEAR(shape27[26], 1.0, 1e-6);
  for (int i = 0; i < 27; ++i)
    if (i != 26)
      EXPECT_NEAR(shape27[i], 0.0, 1e-6);
}

TEST(ShapeFunctions3D, TwentySevenNodeDerivativesSumToZero) {
  auto ders = shape_function_derivatives(0.0, 0.2, 0.0, 27);
  ASSERT_EQ(ders.size(), 3); // ndim = 3
  ASSERT_EQ(ders[0].size(), 27);
  ASSERT_EQ(ders[1].size(), 27);
  ASSERT_EQ(ders[2].size(), 27);
  type_real sum_xi = 0, sum_eta = 0, sum_zeta = 0;
  for (int i = 0; i < 27; ++i) {
    sum_xi += ders[0][i];
    sum_eta += ders[1][i];
    sum_zeta += ders[2][i];
  }
  EXPECT_NEAR(sum_xi, 0.0, 1e-6);
  EXPECT_NEAR(sum_eta, 0.0, 1e-6);
  EXPECT_NEAR(sum_zeta, 0.0, 1e-6);
}
