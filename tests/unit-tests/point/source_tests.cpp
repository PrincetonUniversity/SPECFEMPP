#include "enumerations/interface.hpp"
#include "enumerations/wavefield.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/point/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Test fixture for point source tests
class PointSourceTest : public ::testing::Test {
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

// Test source construction for 2D Acoustic
TEST_F(PointSourceTest, Source2DAcoustic) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::acoustic,
                    wavefield::simulation_field::forward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::acoustic);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::forward);
  constexpr auto components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::components;
  EXPECT_EQ(source_type::components, components);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 1.5 };
  typename source_type::value_type lagrange = { 0.75 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 1.5);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.75);
}

// Test source construction for 2D Elastic
TEST_F(PointSourceTest, Source2DElastic) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::elastic_psv,
                    wavefield::simulation_field::forward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::elastic_psv);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::forward);
  constexpr auto components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::elastic_psv>::components;
  EXPECT_EQ(source_type::components, components);

  // For 2D elastic, components should be 2
  EXPECT_EQ(source_type::components, 2);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 2.0, 3.0 };
  typename source_type::value_type lagrange = { 0.8, 0.9 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 2.0);
  EXPECT_REAL_EQ(src.stf(1), 3.0);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.8);
  EXPECT_REAL_EQ(src.lagrange_interpolant(1), 0.9);
}

// Test source construction for 2D Poroelastic
TEST_F(PointSourceTest, Source2DPoroelastic) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::poroelastic,
                    wavefield::simulation_field::forward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::poroelastic);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::forward);
  constexpr auto components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::poroelastic>::components;
  EXPECT_EQ(source_type::components, components);

  // For 2D poroelastic, components should be 4
  EXPECT_EQ(source_type::components, 4);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 1.0, 2.0, 3.0, 4.0 };
  typename source_type::value_type lagrange = { 0.1, 0.2, 0.3, 0.4 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 1.0);
  EXPECT_REAL_EQ(src.stf(1), 2.0);
  EXPECT_REAL_EQ(src.stf(2), 3.0);
  EXPECT_REAL_EQ(src.stf(3), 4.0);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.1);
  EXPECT_REAL_EQ(src.lagrange_interpolant(1), 0.2);
  EXPECT_REAL_EQ(src.lagrange_interpolant(2), 0.3);
  EXPECT_REAL_EQ(src.lagrange_interpolant(3), 0.4);
}

// Test source construction for 3D Acoustic
TEST_F(PointSourceTest, Source3DAcoustic) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim3, element::medium_tag::acoustic,
                    wavefield::simulation_field::forward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::acoustic);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::forward);
  constexpr auto components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::acoustic>::components;
  EXPECT_EQ(source_type::components, components);

  // For 3D acoustic, components should be 1
  EXPECT_EQ(source_type::components, 1);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 5.5 };
  typename source_type::value_type lagrange = { 0.65 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 5.5);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.65);
}

// Test source construction for 3D Elastic
TEST_F(PointSourceTest, Source3DElastic) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim3, element::medium_tag::elastic,
                    wavefield::simulation_field::forward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::elastic);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::forward);
  constexpr auto components =
      element::attributes<dimension::type::dim3,
                          element::medium_tag::elastic>::components;
  EXPECT_EQ(source_type::components, components);

  // For 3D elastic, components should be 3
  EXPECT_EQ(source_type::components, 3);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 1.1, 2.2, 3.3 };
  typename source_type::value_type lagrange = { 0.11, 0.22, 0.33 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 1.1);
  EXPECT_REAL_EQ(src.stf(1), 2.2);
  EXPECT_REAL_EQ(src.stf(2), 3.3);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.11);
  EXPECT_REAL_EQ(src.lagrange_interpolant(1), 0.22);
  EXPECT_REAL_EQ(src.lagrange_interpolant(2), 0.33);
}

// Test source for adjoint wavefield
TEST_F(PointSourceTest, SourceForAdjoint) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::acoustic,
                    wavefield::simulation_field::adjoint>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::acoustic);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::adjoint);
  constexpr auto components =
      element::attributes<dimension::type::dim2,
                          element::medium_tag::acoustic>::components;
  EXPECT_EQ(source_type::components, components);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 7.7 };
  typename source_type::value_type lagrange = { 0.77 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 7.7);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.77);
}

// Test source for backward wavefield
TEST_F(PointSourceTest, SourceForBackward) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::elastic_psv,
                    wavefield::simulation_field::backward>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::elastic_psv);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::backward);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 8.1, 8.2 };
  typename source_type::value_type lagrange = { 0.81, 0.82 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 8.1);
  EXPECT_REAL_EQ(src.stf(1), 8.2);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.81);
  EXPECT_REAL_EQ(src.lagrange_interpolant(1), 0.82);
}

// Test source for buffer wavefield
TEST_F(PointSourceTest, SourceForBuffer) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::elastic_psv,
                    wavefield::simulation_field::buffer>;

  // Verify static properties
  EXPECT_EQ(source_type::medium_tag, element::medium_tag::elastic_psv);
  EXPECT_EQ(source_type::wavefield_tag, wavefield::simulation_field::buffer);

  // Create STF and Lagrange interpolant values
  typename source_type::value_type stf = { 9.1, 9.2 };
  typename source_type::value_type lagrange = { 0.91, 0.92 };

  // Construct source object
  source_type src(stf, lagrange);

  // Verify values
  EXPECT_REAL_EQ(src.stf(0), 9.1);
  EXPECT_REAL_EQ(src.stf(1), 9.2);
  EXPECT_REAL_EQ(src.lagrange_interpolant(0), 0.91);
  EXPECT_REAL_EQ(src.lagrange_interpolant(1), 0.92);
}

// Test default constructor
TEST_F(PointSourceTest, DefaultConstructor) {
  // Define the source type
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::acoustic,
                    wavefield::simulation_field::forward>;

  // Create source with default constructor
  source_type src;

  // The values should be initialized to zero (or other default)
  // but we can't assert specific values since it's implementation dependent
  // Just verify that the constructor doesn't cause errors
  SUCCEED();
}

// Test accessor base type inheritance
TEST_F(PointSourceTest, AccessorBaseType) {
  using source_type =
      point::source<dimension::type::dim2, element::medium_tag::acoustic,
                    wavefield::simulation_field::forward>;

  // Check if source_type is derived from the correct base class
  bool is_accessor =
      std::is_base_of<specfem::data_access::Accessor<
                          specfem::data_access::AccessorType::point,
                          specfem::data_access::DataClassType::source,
                          dimension::type::dim2, false>,
                      source_type>::value;

  EXPECT_TRUE(is_accessor);
}
