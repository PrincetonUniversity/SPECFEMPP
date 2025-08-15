#include "enumerations/interface.hpp"
#include "specfem/point/jacobian_matrix.hpp"
#include "specfem/point/stress.hpp"
#include "specfem_setup.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Base test fixture for stress tests with template parameter for SIMD
template <bool UseSIMD> class PointStressTestUntyped : public ::testing::Test {
protected:
  // Define SIMD-related types for convenience
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

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
class PointStressTest : public PointStressTestUntyped<T::value> {};

using PointStressTestSerial = PointStressTest<Serial>;

TYPED_TEST_SUITE(PointStressTest, TestTypes);

// Test stress tensor for 2D acoustic medium
TYPED_TEST(PointStressTest, Stress2DAcoustic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype val1{ 1.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val2{ 2.5 };

  // Create a stress tensor for acoustic medium
  // For acoustic medium in 2D, it's a 1x2 tensor (component x dimension)
  typename stress_type::value_type T(val1, val2);

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 0), val1))
      << ExpectedGot(val1, stress.T(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 1), val2))
      << ExpectedGot(val2, stress.T(0, 1));
}

// Test stress tensor for 2D elastic medium
TYPED_TEST(PointStressTest, Stress2DElastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 2D elastic medium
  using stress_type =
      point::stress<dimension::type::dim2, element::medium_tag::elastic_psv,
                    using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype val11{
    1.1
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val21{
    2.1
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val12{
    1.2
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val22{
    2.2
  };

  // Create a stress tensor for elastic medium
  // For elastic medium in 2D, it's a 2x2 tensor (component x dimension)
  typename stress_type::value_type T(val11, val21, val12, val22);

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 0), val11))
      << ExpectedGot(val11, stress.T(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(1, 0), val21))
      << ExpectedGot(val21, stress.T(1, 0));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 1), val12))
      << ExpectedGot(val12, stress.T(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(1, 1), val22))
      << ExpectedGot(val22, stress.T(1, 1));
}

// Test stress tensor for 2D poroelastic medium
TYPED_TEST(PointStressTest, Stress2DPoroelastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 2D poroelastic medium
  using stress_type =
      point::stress<dimension::type::dim2, element::medium_tag::poroelastic,
                    using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype vals[8];

  // Initialize values
  for (int j = 0; j < 8; ++j) {
    vals[j] = { static_cast<type_real>(1.1) +
                (j % 4) * static_cast<type_real>(0.1) +
                (j / 4) * static_cast<type_real>(0.1) };
  }

  // Create a stress tensor for poroelastic medium
  // For poroelastic medium in 2D, it's a 4x2 tensor (component x dimension)
  typename stress_type::value_type T(
      vals[0], vals[1], vals[2], vals[3], // first dimension (column 0)
      vals[4], vals[5], vals[6], vals[7]  // second dimension (column 1)
  );

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      int idx = i + j * 4;
      EXPECT_TRUE(specfem::utilities::is_close(stress.T(i, j), vals[idx]))
          << ExpectedGot(vals[idx], stress.T(i, j)) << " at index (" << i << ","
          << j << ")";
    }
  }
}

// Test stress tensor for 3D acoustic medium
TYPED_TEST(PointStressTest, Stress3DAcoustic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 3D acoustic medium
  using stress_type = point::stress<dimension::type::dim3,
                                    element::medium_tag::acoustic, using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype val1{ 1.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val2{ 2.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val3{ 3.5 };

  // Create a stress tensor for acoustic medium
  // For acoustic medium in 3D, it's a 1x3 tensor (component x dimension)
  typename stress_type::value_type T(val1, val2, val3);

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 0), val1))
      << ExpectedGot(val1, stress.T(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 1), val2))
      << ExpectedGot(val2, stress.T(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 2), val3))
      << ExpectedGot(val3, stress.T(0, 2));
}

// Test stress tensor for 3D elastic medium
TYPED_TEST(PointStressTest, Stress3DElastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 3D elastic medium
  using stress_type = point::stress<dimension::type::dim3,
                                    element::medium_tag::elastic, using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype vals[9];

  // Initialize values
  for (int j = 0; j < 9; ++j) {
    vals[j] = { static_cast<type_real>(1.1) +
                (j % 3) * static_cast<type_real>(0.1) +
                (j / 3) * static_cast<type_real>(0.1) };
  }

  // Create a stress tensor for elastic medium in 3D (3x3)
  typename stress_type::value_type T(vals[0], vals[1], vals[2], // first column
                                     vals[3], vals[4], vals[5], // second column
                                     vals[6], vals[7], vals[8]  // third column
  );

  // Construct stress object
  stress_type stress(T);

  // Verify values with transposed indexing
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int idx = i + j * 3;
      EXPECT_TRUE(specfem::utilities::is_close(stress.T(i, j), vals[idx]))
          << ExpectedGot(vals[idx], stress.T(i, j)) << " at index (" << i << ","
          << j << ")";
    }
  }
}

// Test default constructor
TYPED_TEST(PointStressTest, DefaultConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;

  // Create a zero value for comparison
  typename specfem::datatype::simd<type_real, using_simd>::datatype zero_val{
    0.0
  };

  // Create stress with default constructor
  stress_type stress;

  // The values should be default initialized (to zero)
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 0), zero_val))
      << ExpectedGot(zero_val, stress.T(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(stress.T(0, 1), zero_val))
      << ExpectedGot(zero_val, stress.T(0, 1));
}

// Test stress operator* with Jacobian matrix in 2D
TEST_F(PointStressTestSerial, StressOperatorMultiply2D) {
  constexpr bool using_simd = false;

  // Define the stress type for 2D elastic medium
  using stress_type =
      point::stress<dimension::type::dim2, element::medium_tag::elastic_psv,
                    using_simd>;

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype val11;
  typename specfem::datatype::simd<type_real, using_simd>::datatype val21;
  typename specfem::datatype::simd<type_real, using_simd>::datatype val12;
  typename specfem::datatype::simd<type_real, using_simd>::datatype val22;

  // Initialize values for a stress tensor with transposed indexing
  val11 = 1.0; // T(0,0)
  val21 = 3.0; // T(1,0)
  val12 = 2.0; // T(0,1)
  val22 = 4.0; // T(1,1)

  // Create a stress tensor
  typename stress_type::value_type T(val11, val21, val12, val22);

  // Create Jacobian matrix
  using pd_type =
      point::jacobian_matrix<dimension::type::dim2, true, using_simd>;
  pd_type pd;
  pd.xix = 0.5;      // dx/dxi
  pd.xiz = 0.6;      // dz/dxi
  pd.gammax = 0.7;   // dx/dgamma
  pd.gammaz = 0.8;   // dz/dgamma
  pd.jacobian = 0.5; // Jacobian factor

  // Construct stress object
  stress_type stress(T);

  // Calculate product
  auto F = stress * pd;

  // Expected values for F
  type_real expected_F00 = 0.85; // (1.0*0.5 + 2.0*0.6) * 0.5
  type_real expected_F01 = 1.15; // ((1.0*0.7 + 2.0*0.8) * 0.5) * 0.5
  type_real expected_F10 = 1.95; // ((3.0*0.5 + 4.0*0.6) * 0.5) * 0.5
  type_real expected_F11 = 2.65; // ((3.0*0.7 + 4.0*0.8) * 0.5) * 0.5

  // Verify the calculation with transposed indexing
  EXPECT_TRUE(specfem::utilities::is_close(F(0, 0), expected_F00))
      << ExpectedGot(expected_F00, F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(F(0, 1), expected_F01))
      << ExpectedGot(expected_F01, F(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(F(1, 0), expected_F10))
      << ExpectedGot(expected_F10, F(1, 0));
  EXPECT_TRUE(specfem::utilities::is_close(F(1, 1), expected_F11))
      << ExpectedGot(expected_F11, F(1, 1));
}

// Test stress for acoustic medium with operator* in 2D
TEST_F(PointStressTestSerial, StressOperatorMultiply2DAcoustic) {
  constexpr bool using_simd = false;

  // Define the stress type for 2D acoustic medium
  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;

  // Create a stress tensor with transposed indexing
  typename stress_type::value_type T(5.0, 6.0); // T(0,0)=5.0, T(0,1)=6.0

  // Create Jacobian matrix
  using pd_type =
      point::jacobian_matrix<dimension::type::dim2, true, using_simd>;
  pd_type pd;
  pd.xix = 0.5;      // dx/dxi
  pd.xiz = 0.6;      // dz/dxi
  pd.gammax = 0.7;   // dx/dgamma
  pd.gammaz = 0.8;   // dz/dgamma
  pd.jacobian = 0.5; // Jacobian factor

  // Construct stress object
  stress_type stress(T);

  // Calculate product
  auto F = stress * pd;

  // Expected values for F
  type_real expected_F00 = 3.05; // (5.0*0.5 + 6.0*0.6) * 0.5
  type_real expected_F01 = 4.15; // (5.0*0.7 + 6.0*0.8) * 0.5

  // Verify the calculation with transposed indexing
  EXPECT_TRUE(specfem::utilities::is_close(F(0, 0), expected_F00))
      << ExpectedGot(expected_F00, F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(F(0, 1), expected_F01))
      << ExpectedGot(expected_F01, F(0, 1));
}

// Test accessor base type inheritance
TYPED_TEST(PointStressTest, AccessorBaseType) {
  constexpr bool using_simd = TypeParam::value;

  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;

  // Check if stress_type is derived from the correct base class
  bool is_accessor =
      std::is_base_of<specfem::data_access::Accessor<
                          specfem::data_access::AccessorType::point,
                          specfem::data_access::DataClassType::stress,
                          dimension::type::dim2, using_simd>,
                      stress_type>::value;

  EXPECT_TRUE(is_accessor);
}

// Test SIMD type propagation
TYPED_TEST(PointStressTest, SIMDTypePropagation) {
  constexpr bool using_simd = TypeParam::value;

  using stress_type = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;

  using expected_simd_type = specfem::datatype::simd<type_real, using_simd>;
  using actual_simd_type = typename stress_type::simd;

  bool is_correct_simd_type =
      std::is_same<actual_simd_type, expected_simd_type>::value;
  EXPECT_TRUE(is_correct_simd_type);

  // Also verify the datatype matches
  using expected_datatype = typename expected_simd_type::datatype;
  using actual_datatype = typename actual_simd_type::datatype;

  bool is_correct_datatype =
      std::is_same<actual_datatype, expected_datatype>::value;
  EXPECT_TRUE(is_correct_datatype);
}

// Test tensor shape and dimensions
TYPED_TEST(PointStressTest, TensorShape) {
  constexpr bool using_simd = TypeParam::value;

  // Test various element types and dimensions
  using acoustic_2d = point::stress<dimension::type::dim2,
                                    element::medium_tag::acoustic, using_simd>;
  using elastic_2d =
      point::stress<dimension::type::dim2, element::medium_tag::elastic_psv,
                    using_simd>;
  using acoustic_3d = point::stress<dimension::type::dim3,
                                    element::medium_tag::acoustic, using_simd>;
  using elastic_3d = point::stress<dimension::type::dim3,
                                   element::medium_tag::elastic, using_simd>;
  using poro_2d = point::stress<dimension::type::dim2,
                                element::medium_tag::poroelastic, using_simd>;

  // Verify tensor shapes
  EXPECT_EQ(acoustic_2d::dimension, 2);
  EXPECT_EQ(acoustic_2d::components, 1);

  EXPECT_EQ(elastic_2d::dimension, 2);
  EXPECT_EQ(elastic_2d::components, 2);

  EXPECT_EQ(acoustic_3d::dimension, 3);
  EXPECT_EQ(acoustic_3d::components, 1);

  EXPECT_EQ(elastic_3d::dimension, 3);
  EXPECT_EQ(elastic_3d::components, 3);

  EXPECT_EQ(poro_2d::dimension, 2);
  EXPECT_EQ(poro_2d::components, 4);
}
