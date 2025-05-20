#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

// Base test fixture for Kokkos initialization
class IndexTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos
    Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos
    Kokkos::finalize();
  }
};

// Tests for 2D index

// Test 2D index default constructor
TEST_F(IndexTest, DefaultConstructor2D) {
  // Default constructor
  specfem::point::index<specfem::dimension::type::dim2> idx;

  // Verify static members
  EXPECT_FALSE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim2);
}

// Test 2D index parameterized constructor
TEST_F(IndexTest, ParameterizedConstructor2D) {
  // Parameters
  const int ispec = 5;
  const int iz = 3;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::index<specfem::dimension::type::dim2> idx(ispec, iz, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.ix, ix);

  // Verify static members
  EXPECT_FALSE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim2);
}

// Tests for 2D SIMD index

// Test 2D SIMD index default constructor
TEST_F(IndexTest, DefaultConstructor2D_SIMD) {
  // Default constructor
  specfem::point::simd_index<specfem::dimension::type::dim2> idx;

  // Verify static members
  EXPECT_TRUE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim2);
}

// Test 2D SIMD index parameterized constructor
TEST_F(IndexTest, ParameterizedConstructor2D_SIMD) {
  // Parameters
  const int ispec = 5;
  const int number_elements = 4;
  const int iz = 3;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim2> idx(
      ispec, number_elements, iz, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.number_elements, number_elements);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.ix, ix);

  // Verify static members
  EXPECT_TRUE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim2);
}

// Test 2D SIMD index mask function
TEST_F(IndexTest, MaskFunction2D_SIMD) {
  // Parameters
  const int ispec = 5;
  const int number_elements = 4;
  const int iz = 3;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim2> idx(
      ispec, number_elements, iz, ix);

  // Test mask function
  EXPECT_TRUE(idx.mask(0));  // Lane 0 is within the number_elements
  EXPECT_TRUE(idx.mask(3));  // Lane 3 is within the number_elements
  EXPECT_FALSE(idx.mask(4)); // Lane 4 is outside the number_elements
  EXPECT_FALSE(idx.mask(5)); // Lane 5 is outside the number_elements
}

// Tests for 3D index

// Test 3D index default constructor
TEST_F(IndexTest, DefaultConstructor3D) {
  // Default constructor
  specfem::point::index<specfem::dimension::type::dim3> idx;

  // Verify static members
  EXPECT_FALSE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim3);
}

// Test 3D index parameterized constructor
TEST_F(IndexTest, ParameterizedConstructor3D) {
  // Parameters
  const int ispec = 5;
  const int iz = 3;
  const int iy = 4;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::index<specfem::dimension::type::dim3> idx(ispec, iz, iy, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.iy, iy);
  EXPECT_EQ(idx.ix, ix);

  // Verify static members
  EXPECT_FALSE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim3);
}

// Tests for 3D SIMD index

// Test 3D SIMD index default constructor
TEST_F(IndexTest, DefaultConstructor3D_SIMD) {
  // Default constructor
  specfem::point::simd_index<specfem::dimension::type::dim3> idx;

  // Verify static members
  EXPECT_TRUE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim3);
}

// Test 3D SIMD index parameterized constructor
TEST_F(IndexTest, ParameterizedConstructor3D_SIMD) {
  // Parameters
  const int ispec = 5;
  const int number_elements = 4;
  const int iz = 3;
  const int iy = 4;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim3> idx(
      ispec, number_elements, iz, iy, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.number_elements, number_elements);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.iy, iy);
  EXPECT_EQ(idx.ix, ix);

  // Verify static members
  EXPECT_TRUE(idx.using_simd);
  EXPECT_EQ(idx.dimension, specfem::dimension::type::dim3);
}

// Test 3D SIMD index mask function
TEST_F(IndexTest, MaskFunction3D_SIMD) {
  // Parameters
  const int ispec = 5;
  const int number_elements = 4;
  const int iz = 3;
  const int iy = 4;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim3> idx(
      ispec, number_elements, iz, iy, ix);

  // Test mask function
  EXPECT_TRUE(idx.mask(0));   // Lane 0 is within the number_elements
  EXPECT_TRUE(idx.mask(3));   // Lane 3 is within the number_elements
  EXPECT_FALSE(idx.mask(4));  // Lane 4 is outside the number_elements
  EXPECT_FALSE(idx.mask(10)); // Lane 10 is outside the number_elements
}

// Test negative indices 2D
TEST_F(IndexTest, NegativeIndices2D) {
  // Parameters with negative values
  const int ispec = -1;
  const int iz = -2;
  const int ix = -3;

  // Constructor with parameters
  specfem::point::index<specfem::dimension::type::dim2> idx(ispec, iz, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.ix, ix);
}

TEST_F(IndexTest, NegativeIndices3D) {
  // Parameters with negative values
  const int ispec = -1;
  const int iz = -2;
  const int iy = -3;
  const int ix = -4;

  // Constructor with parameters
  specfem::point::index<specfem::dimension::type::dim3> idx(ispec, iz, iy, ix);

  // Check values
  EXPECT_EQ(idx.ispec, ispec);
  EXPECT_EQ(idx.iz, iz);
  EXPECT_EQ(idx.iy, iy);
  EXPECT_EQ(idx.ix, ix);
}

// Test edge cases for SIMD number_elements
TEST_F(IndexTest, ZeroElements2D_SIMD) {
  // Parameters with zero elements
  const int ispec = 5;
  const int number_elements = 0;
  const int iz = 3;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim2> idx(
      ispec, number_elements, iz, ix);

  // Check values
  EXPECT_EQ(idx.number_elements, number_elements);

  // Test mask function with zero elements
  EXPECT_FALSE(idx.mask(0)); // All lanes should be outside when number_elements
                             // is 0
}

TEST_F(IndexTest, ZeroElements3D) {
  // Parameters with zero elements
  const int ispec = 5;
  const int number_elements = 0;
  const int iz = 3;
  const int iy = 4;
  const int ix = 2;

  // Constructor with parameters
  specfem::point::simd_index<specfem::dimension::type::dim3> idx(
      ispec, number_elements, iz, iy, ix);

  // Check values
  EXPECT_EQ(idx.number_elements, number_elements);

  // Test mask function with zero elements
  EXPECT_FALSE(idx.mask(0)); // All lanes should be outside when number_elements
                             // is 0
}

// Main function
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
