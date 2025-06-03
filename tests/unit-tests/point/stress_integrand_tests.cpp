#include "enumerations/interface.hpp"
#include "specfem/point/stress_integrand.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Test fixture for stress integrand tests
class PointStressIntegrandTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos if needed for tests
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos if needed
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};

// Test stress_integrand for 2D acoustic medium
TEST_F(PointStressIntegrandTest, StressIntegrand2DAcoustic) {
  // Define the stress_integrand type for 2D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::components;

  EXPECT_EQ(stress_integrand_type::dimension, expected_dimension);
  EXPECT_EQ(stress_integrand_type::components, expected_components);
  EXPECT_EQ(stress_integrand_type::dimension, 2);
  EXPECT_EQ(stress_integrand_type::components, 1);

  // Create a stress integrand tensor for acoustic medium
  // For acoustic medium in 2D, it's a 1x2 tensor (component x dimension)
  // With transposed indexing, we need to be careful about the order of
  // initialization

  // APPROACH 1: Initialize with individual values in column-major order
  typename stress_integrand_type::value_type F(1.5, 2.5);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values - if transposed, we might be accessing them differently
  EXPECT_REAL_EQ(si.F(0, 0), 1.5);
  EXPECT_REAL_EQ(si.F(0, 1), 2.5);
}

// Test stress_integrand for 2D elastic medium
TEST_F(PointStressIntegrandTest, StressIntegrand2DElastic) {
  // Define the stress_integrand type for 2D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::elastic_psv, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::elastic_psv>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::elastic_psv>::components;

  EXPECT_EQ(stress_integrand_type::dimension, expected_dimension);
  EXPECT_EQ(stress_integrand_type::components, expected_components);
  EXPECT_EQ(stress_integrand_type::dimension, 2);
  EXPECT_EQ(stress_integrand_type::components, 2);

  // Create a stress integrand tensor for elastic medium
  // For elastic medium in 2D, it's a 2x2 tensor (component x dimension)
  // If indices are transposed, we initialize in column-major order: (0,0),
  // (1,0), (0,1), (1,1)
  typename stress_integrand_type::value_type F(1.1, 2.1, 1.2, 2.2);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values - if transposed, the indices would be different
  EXPECT_REAL_EQ(si.F(0, 0), 1.1);
  EXPECT_REAL_EQ(si.F(1, 0), 2.1);
  EXPECT_REAL_EQ(si.F(0, 1), 1.2);
  EXPECT_REAL_EQ(si.F(1, 1), 2.2);
}

// Test stress_integrand for 2D poroelastic medium
TEST_F(PointStressIntegrandTest, StressIntegrand2DPoroelastic) {
  // Define the stress_integrand type for 2D poroelastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::poroelastic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::poroelastic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::poroelastic>::components;

  EXPECT_EQ(stress_integrand_type::dimension, expected_dimension);
  EXPECT_EQ(stress_integrand_type::components, expected_components);
  EXPECT_EQ(stress_integrand_type::dimension, 2);
  EXPECT_EQ(stress_integrand_type::components, 4);

  // Create a stress integrand tensor for poroelastic medium
  // For poroelastic medium in 2D, it's a 4x2 tensor (component x dimension)
  // With transposed indexing, use column-major order:
  // (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
  typename stress_integrand_type::value_type F(
      1.1, 2.1, 3.1, 4.1, // first dimension (column 0)
      1.2, 2.2, 3.2, 4.2  // second dimension (column 1)
  );

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(si.F(0, 0), 1.1);
  EXPECT_REAL_EQ(si.F(1, 0), 2.1);
  EXPECT_REAL_EQ(si.F(2, 0), 3.1);
  EXPECT_REAL_EQ(si.F(3, 0), 4.1);
  EXPECT_REAL_EQ(si.F(0, 1), 1.2);
  EXPECT_REAL_EQ(si.F(1, 1), 2.2);
  EXPECT_REAL_EQ(si.F(2, 1), 3.2);
  EXPECT_REAL_EQ(si.F(3, 1), 4.2);
}

// Test stress_integrand for 3D acoustic medium
TEST_F(PointStressIntegrandTest, StressIntegrand3DAcoustic) {
  // Define the stress_integrand type for 3D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::acoustic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::acoustic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::acoustic>::components;

  EXPECT_EQ(stress_integrand_type::dimension, expected_dimension);
  EXPECT_EQ(stress_integrand_type::components, expected_components);
  EXPECT_EQ(stress_integrand_type::dimension, 3);
  EXPECT_EQ(stress_integrand_type::components, 1);

  // Create a stress integrand tensor for acoustic medium
  // For acoustic medium in 3D, it's a 1x3 tensor (component x dimension)
  // With transposed indexing, column-major order still works the same for 1
  // component
  typename stress_integrand_type::value_type F(1.5, 2.5, 3.5);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values
  EXPECT_REAL_EQ(si.F(0, 0), 1.5);
  EXPECT_REAL_EQ(si.F(0, 1), 2.5);
  EXPECT_REAL_EQ(si.F(0, 2), 3.5);
}

// Test stress_integrand for 3D elastic medium
TEST_F(PointStressIntegrandTest, StressIntegrand3DElastic) {
  // Define the stress_integrand type for 3D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::elastic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::elastic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::elastic>::components;

  EXPECT_EQ(stress_integrand_type::dimension, expected_dimension);
  EXPECT_EQ(stress_integrand_type::components, expected_components);
  EXPECT_EQ(stress_integrand_type::dimension, 3);
  EXPECT_EQ(stress_integrand_type::components, 3);

  // Create a stress integrand tensor for elastic medium
  // For elastic medium in 3D, it's a 3x3 tensor (component x dimension)
  // With transposed indexing (column-major), we initialize as:
  // (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
  typename stress_integrand_type::value_type F(
      1.1, 2.1, 3.1, // first dimension (column 0)
      1.2, 2.2, 3.2, // second dimension (column 1)
      1.3, 2.3, 3.3  // third dimension (column 2)
  );

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(si.F(0, 0), 1.1);
  EXPECT_REAL_EQ(si.F(1, 0), 2.1);
  EXPECT_REAL_EQ(si.F(2, 0), 3.1);
  EXPECT_REAL_EQ(si.F(0, 1), 1.2);
  EXPECT_REAL_EQ(si.F(1, 1), 2.2);
  EXPECT_REAL_EQ(si.F(2, 1), 3.2);
  EXPECT_REAL_EQ(si.F(0, 2), 1.3);
  EXPECT_REAL_EQ(si.F(1, 2), 2.3);
  EXPECT_REAL_EQ(si.F(2, 2), 3.3);
}

// Test default constructor
TEST_F(PointStressIntegrandTest, DefaultConstructor) {
  // Define the stress_integrand type for 2D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, false>;

  // Create stress_integrand with default constructor
  stress_integrand_type si;

  // The values should be default initialized (to zero based on RegisterArray
  // implementation)
  EXPECT_REAL_EQ(si.F(0, 0), 0.0);
  EXPECT_REAL_EQ(si.F(0, 1), 0.0);
}

// Test constructor with uniform value
TEST_F(PointStressIntegrandTest, ConstantConstructor) {
  // Define the stress_integrand type for 2D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::elastic_psv, false>;

  // Create a stress integrand tensor initialized with a constant value
  typename stress_integrand_type::value_type F(3.14);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify all values are set to the constant
  EXPECT_REAL_EQ(si.F(0, 0), 3.14);
  EXPECT_REAL_EQ(si.F(0, 1), 3.14);
  EXPECT_REAL_EQ(si.F(1, 0), 3.14);
  EXPECT_REAL_EQ(si.F(1, 1), 3.14);
}

// Test SIMD version of stress_integrand
TEST_F(PointStressIntegrandTest, StressIntegrand2DAcoustic_SIMD) {
  // Define the SIMD stress_integrand type for 2D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, true>;

  // Verify SIMD flag is propagated to the base class
  using base_type =
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::stress_integrand,
                                  dimension::type::dim2, true>;

  bool is_simd_accessor =
      std::is_base_of<base_type, stress_integrand_type>::value;
  EXPECT_TRUE(is_simd_accessor);

  // Verify static properties for SIMD version
  EXPECT_EQ(stress_integrand_type::dimension, 2);
  EXPECT_EQ(stress_integrand_type::components, 1);

  // Create a stress integrand object (default initialized)
  stress_integrand_type si;

  // Just verify the type can be instantiated
  SUCCEED();
}

// Test accessor base type inheritance
TEST_F(PointStressIntegrandTest, AccessorBaseType) {
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, false>;

  // Check if stress_integrand_type is derived from the correct base class
  bool is_accessor = std::is_base_of<
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::stress_integrand,
                                  dimension::type::dim2, false>,
      stress_integrand_type>::value;

  EXPECT_TRUE(is_accessor);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
