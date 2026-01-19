#include "enumerations/interface.hpp"
#include "specfem/point/stress_integrand.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Base test fixture for stress integrand tests with template parameter for SIMD
template <bool UseSIMD>
class PointStressIntegrandTestUntyped : public ::testing::Test {
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
class PointStressIntegrandTest
    : public PointStressIntegrandTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointStressIntegrandTest, TestTypes);

// Test stress_integrand for 2D acoustic medium
TYPED_TEST(PointStressIntegrandTest, StressIntegrand2DAcoustic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 2D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, using_simd>;

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

  // Create test values for the tensor
  typename specfem::datatype::simd<type_real, using_simd>::datatype val1{ 1.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val2{ 2.5 };

  // Create a stress integrand tensor for acoustic medium
  // For acoustic medium in 2D, it's a 1x2 tensor (component x dimension)
  typename stress_integrand_type::value_type F(val1, val2);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values with is_close
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 0), val1))
      << ExpectedGot(val1, si.F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 1), val2))
      << ExpectedGot(val2, si.F(0, 1));
}

// Test stress_integrand for 2D elastic medium
TYPED_TEST(PointStressIntegrandTest, StressIntegrand2DElastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 2D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::elastic_psv, using_simd>;

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

  // Create test values for the tensor
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

  // Create a stress integrand tensor for elastic medium
  // For elastic medium in 2D, it's a 2x2 tensor (component x dimension)
  typename stress_integrand_type::value_type F(val11, val21, val12, val22);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values using is_close
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 0), val11))
      << ExpectedGot(val11, si.F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(1, 0), val21))
      << ExpectedGot(val21, si.F(1, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 1), val12))
      << ExpectedGot(val12, si.F(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(1, 1), val22))
      << ExpectedGot(val22, si.F(1, 1));
}

// Test stress_integrand for 2D poroelastic medium
TYPED_TEST(PointStressIntegrandTest, StressIntegrand2DPoroelastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 2D poroelastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::poroelastic, using_simd>;

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

  // Create test values for tensor elements
  typename specfem::datatype::simd<type_real, using_simd>::datatype
      vals[8]; // 4 components × 2 dimensions

  // Initialize values
  for (int j = 0; j < 8; ++j) {
    vals[j] = { static_cast<type_real>(1.0) + j * static_cast<type_real>(0.1) +
                (j / 4) * static_cast<type_real>(0.1) };
  }

  // Create tensor for poroelastic medium (4x2)
  typename stress_integrand_type::value_type F(
      vals[0], vals[1], vals[2], vals[3], // first dimension (column 0)
      vals[4], vals[5], vals[6], vals[7]  // second dimension (column 1)
  );

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      int idx = i + j * 4;
      EXPECT_TRUE(specfem::utilities::is_close(si.F(i, j), vals[idx]))
          << ExpectedGot(vals[idx], si.F(i, j)) << " at index (" << i << ","
          << j << ")";
    }
  }
}

// Test stress_integrand for 3D acoustic medium
TYPED_TEST(PointStressIntegrandTest, StressIntegrand3DAcoustic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 3D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::acoustic, using_simd>;

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

  // Create test values
  typename specfem::datatype::simd<type_real, using_simd>::datatype val1{ 1.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val2{ 2.5 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype val3{ 3.5 };

  // Create tensor for acoustic medium in 3D (1x3)
  typename stress_integrand_type::value_type F(val1, val2, val3);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 0), val1))
      << ExpectedGot(val1, si.F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 1), val2))
      << ExpectedGot(val2, si.F(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 2), val3))
      << ExpectedGot(val3, si.F(0, 2));
}

// Test stress_integrand for 3D elastic medium
TYPED_TEST(PointStressIntegrandTest, StressIntegrand3DElastic) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 3D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::elastic, using_simd>;

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

  // Create test values for 3×3 tensor
  typename specfem::datatype::simd<type_real, using_simd>::datatype vals[9];

  // Initialize values
  for (int j = 0; j < 9; ++j) {
    vals[j] = { static_cast<type_real>(1.0) +
                (j % 3) * static_cast<type_real>(0.1) +
                (j / 3) * static_cast<type_real>(0.1) };
  }

  // Create tensor for elastic medium in 3D (3×3)
  typename stress_integrand_type::value_type F(
      vals[0], vals[1], vals[2], // first column
      vals[3], vals[4], vals[5], // second column
      vals[6], vals[7], vals[8]  // third column
  );

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify values
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int idx = i + j * 3;
      EXPECT_TRUE(specfem::utilities::is_close(si.F(i, j), vals[idx]))
          << ExpectedGot(vals[idx], si.F(i, j)) << " at index (" << i << ","
          << j << ")";
    }
  }
}

// Test default constructor
TYPED_TEST(PointStressIntegrandTest, DefaultConstructor) {
  constexpr bool using_simd = TypeParam::value;

  // Define the stress_integrand type for 2D acoustic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, using_simd>;

  // Create stress_integrand with default constructor
  stress_integrand_type si;

  // Get a zero value in appropriate type
  typename specfem::datatype::simd<type_real, using_simd>::datatype zero_val{
    0.0
  };

  // The values should be default initialized (to zero)
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 0), zero_val))
      << ExpectedGot(zero_val, si.F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 1), zero_val))
      << ExpectedGot(zero_val, si.F(0, 1));
}

// Test constructor with uniform value
TYPED_TEST(PointStressIntegrandTest, ConstantConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Define the stress_integrand type for 2D elastic medium
  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::elastic_psv, using_simd>;

  // Create a constant value
  typename specfem::datatype::simd<type_real, using_simd>::datatype const_val{
    3.14
  };

  // Create a stress integrand tensor initialized with a constant value
  typename stress_integrand_type::value_type F(const_val);

  // Construct stress_integrand object
  stress_integrand_type si(F);

  // Verify all values are set to the constant
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 0), const_val))
      << ExpectedGot(const_val, si.F(0, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(0, 1), const_val))
      << ExpectedGot(const_val, si.F(0, 1));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(1, 0), const_val))
      << ExpectedGot(const_val, si.F(1, 0));
  EXPECT_TRUE(specfem::utilities::is_close(si.F(1, 1), const_val))
      << ExpectedGot(const_val, si.F(1, 1));
}

// Test accessor base type inheritance
TYPED_TEST(PointStressIntegrandTest, AccessorBaseType) {
  constexpr bool using_simd = TypeParam::value;

  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, using_simd>;

  // Check if stress_integrand_type is derived from the correct base class
  bool is_accessor =
      std::is_base_of<specfem::data_access::Accessor<
                          specfem::data_access::AccessorType::point,
                          specfem::data_access::DataClassType::stress_integrand,
                          dimension::type::dim2, using_simd>,
                      stress_integrand_type>::value;

  EXPECT_TRUE(is_accessor);
}

// Test tensor shape and dimensions
TYPED_TEST(PointStressIntegrandTest, TensorShape) {
  constexpr bool using_simd = TypeParam::value;

  // Test various element types and dimensions
  using acoustic_2d =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, using_simd>;
  using elastic_2d =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::elastic_psv, using_simd>;
  using acoustic_3d =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::acoustic, using_simd>;
  using elastic_3d =
      point::stress_integrand<dimension::type::dim3,
                              element::medium_tag::elastic, using_simd>;
  using poro_2d =
      point::stress_integrand<dimension::type::dim2,
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

// Test SIMD type propagation
TYPED_TEST(PointStressIntegrandTest, SIMDTypePropagation) {
  constexpr bool using_simd = TypeParam::value;

  using stress_integrand_type =
      point::stress_integrand<dimension::type::dim2,
                              element::medium_tag::acoustic, using_simd>;

  using expected_simd_type = specfem::datatype::simd<type_real, using_simd>;
  using actual_simd_type = typename stress_integrand_type::simd;

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
