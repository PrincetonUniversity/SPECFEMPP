#include "enumerations/interface.hpp"
#include "specfem/point/partial_derivatives.hpp"
#include "specfem/point/stress.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Test fixture for stress tensor tests
class PointStressTest : public ::testing::Test {
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

// Test stress tensor for 2D acoustic medium
TEST_F(PointStressTest, Stress2DAcoustic) {
  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::components;

  EXPECT_EQ(stress_type::dimension, expected_dimension);
  EXPECT_EQ(stress_type::components, expected_components);
  EXPECT_EQ(stress_type::dimension, 2);
  EXPECT_EQ(stress_type::components, 1);

  // Create a stress tensor for acoustic medium
  // For acoustic medium in 2D, it's a 1x2 tensor (component x dimension)
  // Using column-major order for transposed indexing: (0,0), (0,1)
  typename stress_type::value_type T(1.5, 2.5);

  // Construct stress object
  stress_type stress(T);

  // Verify values - with transposed indexing, these values should match our
  // initialization
  EXPECT_REAL_EQ(stress.T(0, 0), 1.5);
  EXPECT_REAL_EQ(stress.T(0, 1), 2.5);
}

// Test stress tensor for 2D elastic medium
TEST_F(PointStressTest, Stress2DElastic) {
  // Define the stress type for 2D elastic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::elastic_psv, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::elastic_psv>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::elastic_psv>::components;

  EXPECT_EQ(stress_type::dimension, expected_dimension);
  EXPECT_EQ(stress_type::components, expected_components);
  EXPECT_EQ(stress_type::dimension, 2);
  EXPECT_EQ(stress_type::components, 2);

  // Create a stress tensor for elastic medium
  // For elastic medium in 2D, it's a 2x2 tensor (component x dimension)
  // Using column-major order for transposed indexing: (0,0), (1,0), (0,1),
  // (1,1)
  typename stress_type::value_type T(1.1, 2.1, 1.2, 2.2);

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(stress.T(0, 0), 1.1);
  EXPECT_REAL_EQ(stress.T(1, 0), 2.1);
  EXPECT_REAL_EQ(stress.T(0, 1), 1.2);
  EXPECT_REAL_EQ(stress.T(1, 1), 2.2);
}

// Test stress tensor for 2D poroelastic medium
TEST_F(PointStressTest, Stress2DPoroelastic) {
  // Define the stress type for 2D poroelastic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::poroelastic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::poroelastic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::poroelastic>::components;

  EXPECT_EQ(stress_type::dimension, expected_dimension);
  EXPECT_EQ(stress_type::components, expected_components);
  EXPECT_EQ(stress_type::dimension, 2);
  EXPECT_EQ(stress_type::components, 4);

  // Create a stress tensor for poroelastic medium
  // For poroelastic medium in 2D, it's a 4x2 tensor (component x dimension)
  // Using column-major order for transposed indexing:
  // (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
  typename stress_type::value_type T(
      1.1, 2.1, 3.1, 4.1, // first dimension (column 0)
      1.2, 2.2, 3.2, 4.2  // second dimension (column 1)
  );

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(stress.T(0, 0), 1.1);
  EXPECT_REAL_EQ(stress.T(1, 0), 2.1);
  EXPECT_REAL_EQ(stress.T(2, 0), 3.1);
  EXPECT_REAL_EQ(stress.T(3, 0), 4.1);
  EXPECT_REAL_EQ(stress.T(0, 1), 1.2);
  EXPECT_REAL_EQ(stress.T(1, 1), 2.2);
  EXPECT_REAL_EQ(stress.T(2, 1), 3.2);
  EXPECT_REAL_EQ(stress.T(3, 1), 4.2);
}

// Test stress tensor for 3D acoustic medium
TEST_F(PointStressTest, Stress3DAcoustic) {
  // Define the stress type for 3D acoustic medium
  using stress_type = point::stress<dimension::type::dim3,
                                    element::medium_tag::acoustic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::acoustic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::acoustic>::components;

  EXPECT_EQ(stress_type::dimension, expected_dimension);
  EXPECT_EQ(stress_type::components, expected_components);
  EXPECT_EQ(stress_type::dimension, 3);
  EXPECT_EQ(stress_type::components, 1);

  // Create a stress tensor for acoustic medium
  // For acoustic medium in 3D, it's a 1x3 tensor (component x dimension)
  // Using column-major ordering
  typename stress_type::value_type T(1.5, 2.5, 3.5);

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(stress.T(0, 0), 1.5);
  EXPECT_REAL_EQ(stress.T(0, 1), 2.5);
  EXPECT_REAL_EQ(stress.T(0, 2), 3.5);
}

// Test stress tensor for 3D elastic medium
TEST_F(PointStressTest, Stress3DElastic) {
  // Define the stress type for 3D elastic medium
  using stress_type =
      point::stress<dimension::type::dim3, element::medium_tag::elastic, false>;

  // Verify static properties
  constexpr int expected_dimension =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::elastic>::dimension;
  constexpr int expected_components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::elastic>::components;

  EXPECT_EQ(stress_type::dimension, expected_dimension);
  EXPECT_EQ(stress_type::components, expected_components);
  EXPECT_EQ(stress_type::dimension, 3);
  EXPECT_EQ(stress_type::components, 3);

  // Create a stress tensor for elastic medium
  // For elastic medium in 3D, it's a 3x3 tensor (component x dimension)
  // Using column-major order for transposed indexing:
  // (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
  typename stress_type::value_type T(
      1.1, 2.1, 3.1, // first dimension (column 0)
      1.2, 2.2, 3.2, // second dimension (column 1)
      1.3, 2.3, 3.3  // third dimension (column 2)
  );

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_REAL_EQ(stress.T(0, 0), 1.1);
  EXPECT_REAL_EQ(stress.T(1, 0), 2.1);
  EXPECT_REAL_EQ(stress.T(2, 0), 3.1);
  EXPECT_REAL_EQ(stress.T(0, 1), 1.2);
  EXPECT_REAL_EQ(stress.T(1, 1), 2.2);
  EXPECT_REAL_EQ(stress.T(2, 1), 3.2);
  EXPECT_REAL_EQ(stress.T(0, 2), 1.3);
  EXPECT_REAL_EQ(stress.T(1, 2), 2.3);
  EXPECT_REAL_EQ(stress.T(2, 2), 3.3);
}

// Test default constructor
TEST_F(PointStressTest, DefaultConstructor) {
  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, false>;

  // Create stress with default constructor
  stress_type stress;

  // The values should be default initialized (to zero based on RegisterArray
  // implementation)
  EXPECT_REAL_EQ(stress.T(0, 0), 0.0);
  EXPECT_REAL_EQ(stress.T(0, 1), 0.0);
}

// Test stress operator* with partial derivatives in 2D
TEST_F(PointStressTest, StressOperatorMultiply2D) {
  // Define the stress type for 2D elastic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::elastic_psv, false>;

  // Create a stress tensor with transposed indexing
  // Note: The operator* implementation expects the transposed indexing pattern
  // T(component, dimension) should be initialized in column-major order
  typename stress_type::value_type T(
      1.0, 3.0, // First column: T(0,0)=1.0, T(1,0)=3.0
      2.0, 4.0  // Second column: T(0,1)=2.0, T(1,1)=4.0
  );

  // Create partial derivatives
  using pd_type =
      point::partial_derivatives<dimension::type::dim2, false, false>;
  pd_type pd;
  pd.xix = 0.5;    // dx/dxi
  pd.xiz = 0.6;    // dz/dxi
  pd.gammax = 0.7; // dx/dgamma
  pd.gammaz = 0.8; // dz/dgamma

  // Construct stress object
  stress_type stress(T);

  // Calculate product
  auto F = stress * pd;

  // Verify the calculation with transposed indexing
  // For component 0:
  // F(0,0) = T(0,0)*xix + T(0,1)*xiz = 1.0*0.5 + 2.0*0.6 = 0.5 + 1.2 = 1.7
  // F(0,1) = T(0,0)*gammax + T(0,1)*gammaz = 1.0*0.7 + 2.0*0.8 = 0.7 + 1.6
  // = 2.3
  EXPECT_REAL_EQ(F(0, 0), 1.7);
  EXPECT_REAL_EQ(F(0, 1), 2.3);

  // For component 1:
  // F(1,0) = T(1,0)*xix + T(1,1)*xiz = 3.0*0.5 + 4.0*0.6 = 1.5 + 2.4 = 3.9
  // F(1,1) = T(1,0)*gammax + T(1,1)*gammaz = 3.0*0.7 + 4.0*0.8 = 2.1 + 3.2
  // = 5.3
  EXPECT_REAL_EQ(F(1, 0), 3.9);
  EXPECT_REAL_EQ(F(1, 1), 5.3);
}

// Test stress for acoustic medium with operator* in 2D
TEST_F(PointStressTest, StressOperatorMultiply2DAcoustic) {
  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, false>;

  // Create a stress tensor with transposed indexing
  // T(0,0)=5.0, T(0,1)=6.0
  typename stress_type::value_type T(5.0, 6.0);

  // Create partial derivatives
  using pd_type =
      point::partial_derivatives<dimension::type::dim2, false, false>;
  pd_type pd;
  pd.xix = 0.5;    // dx/dxi
  pd.xiz = 0.6;    // dz/dxi
  pd.gammax = 0.7; // dx/dgamma
  pd.gammaz = 0.8; // dz/dgamma

  // Construct stress object
  stress_type stress(T);

  // Calculate product
  auto F = stress * pd;

  // Verify the calculation with transposed indexing
  // For acoustic medium, there's only one component:
  // F(0,0) = T(0,0)*xix + T(0,1)*xiz = 5.0*0.5 + 6.0*0.6 = 2.5 + 3.6 = 6.1
  // F(0,1) = T(0,0)*gammax + T(0,1)*gammaz = 5.0*0.7 + 6.0*0.8 = 3.5 + 4.8
  // = 8.3
  EXPECT_REAL_EQ(F(0, 0), 6.1);
  EXPECT_REAL_EQ(F(0, 1), 8.3);
}

// Test SIMD version of stress
TEST_F(PointStressTest, Stress2DAcoustic_SIMD) {
  // Define the SIMD stress type for 2D acoustic medium
  using stress_type =
      point::stress<dimension::type::dim2, element::medium_tag::acoustic, true>;

  // Verify SIMD flag is propagated to the base class
  using base_type =
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::stress,
                                  dimension::type::dim2, true>;

  bool is_simd_accessor = std::is_base_of<base_type, stress_type>::value;
  EXPECT_TRUE(is_simd_accessor);

  // Verify static properties for SIMD version
  EXPECT_EQ(stress_type::dimension, 2);
  EXPECT_EQ(stress_type::components, 1);

  // Create a stress object (default initialized)
  stress_type stress;

  // Just verify the type can be instantiated
  SUCCEED();
}

// Test accessor base type inheritance
TEST_F(PointStressTest, AccessorBaseType) {
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, false>;

  // Check if stress_type is derived from the correct base class
  bool is_accessor = std::is_base_of<
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::stress,
                                  dimension::type::dim2, false>,
      stress_type>::value;

  EXPECT_TRUE(is_accessor);
}
