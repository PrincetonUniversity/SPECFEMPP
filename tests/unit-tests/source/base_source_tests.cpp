#include "source_time_function/interface.hpp"
#include "specfem/point/coordinates.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Test base class get_coords() method through concrete implementation
TEST(SOURCES_BASE, BaseSource_GetCoords_ReturnsCorrectArray) {
  type_real x = 5.5;
  type_real z = -2.3;

  // Create concrete source instance (using force as concrete implementation)
  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0, // x, z, angle
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  // Access through base class reference to test base class functionality
  specfem::sources::source<specfem::dimension::type::dim2> &base_source =
      force_source;

  // Test the base class get_coords() method
  auto coords = base_source.get_coords();

  // Verify the returned array has correct values
  EXPECT_REAL_EQ(coords[0], x);
  EXPECT_REAL_EQ(coords[1], z);
}

// Test get_coords() with zero coordinates
TEST(SOURCES_BASE, GetCoords_ZeroCoordinates) {
  type_real x = 0.0;
  type_real z = 0.0;

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords = force_source.get_coords();

  EXPECT_REAL_EQ(coords[0], 0.0);
  EXPECT_REAL_EQ(coords[1], 0.0);
}

// Test get_coords() with negative coordinates
TEST(SOURCES_BASE, GetCoords_NegativeCoordinates) {
  type_real x = -10.5;
  type_real z = -3.7;

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords = force_source.get_coords();

  EXPECT_REAL_EQ(coords[0], x);
  EXPECT_REAL_EQ(coords[1], z);
}

// Test get_coords() with extreme values
TEST(SOURCES_BASE, GetCoords_ExtremeValues) {
  type_real x = 1e10;
  type_real z = -1e10;

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords = force_source.get_coords();

  EXPECT_REAL_EQ(coords[0], x);
  EXPECT_REAL_EQ(coords[1], z);
}

// Test that get_coords() works with different source types
TEST(SOURCES_BASE, GetCoords_DifferentSourceTypes) {
  type_real x = 3.14;
  type_real z = 2.71;

  // Test with moment tensor source
  specfem::sources::moment_tensor<specfem::dimension::type::dim2>
      moment_tensor_source(x, z, 1.0, 1.0, 0.0, // x, z, Mxx, Mzz, Mxz
                           std::make_unique<specfem::forcing_function::Ricker>(
                               10, 0.01, 1.0, 0.0, 1.0, false),
                           specfem::wavefield::simulation_field::forward);

  auto coords_tensor = moment_tensor_source.get_coords();
  EXPECT_REAL_EQ(coords_tensor[0], x);
  EXPECT_REAL_EQ(coords_tensor[1], z);

  // Test with external source
  specfem::sources::external<specfem::dimension::type::dim2> external_source(
      x, z,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords_external = external_source.get_coords();
  EXPECT_REAL_EQ(coords_external[0], x);
  EXPECT_REAL_EQ(coords_external[1], z);
}

// Test that get_coords() returns correct size array
TEST(SOURCES_BASE, GetCoords_CorrectArraySize) {
  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      1.0, 2.0, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords = force_source.get_coords();

  // For 2D, should have exactly 2 elements
  EXPECT_EQ(coords.extent(0), 2);
}

// Test consistency between get_x()/get_z() and get_coords() through base class
TEST(SOURCES_BASE, GetCoords_ConsistentWithGetters) {
  type_real x = 7.5;
  type_real z = -4.2;

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  // Access through base class reference
  specfem::sources::source<specfem::dimension::type::dim2> &base_source =
      force_source;

  auto coords = base_source.get_coords();

  // Verify consistency with existing getters through base class
  EXPECT_REAL_EQ(coords[0], base_source.get_x());
  EXPECT_REAL_EQ(coords[1], base_source.get_z());
}

// Test that the returned array can be used with our new coordinate constructors
TEST(SOURCES_BASE, GetCoords_CompatibleWithCoordinateConstructors) {
  type_real x = 1.5;
  type_real z = 3.7;

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  auto coords = force_source.get_coords();

  // Test that we can create global_coordinates from the returned array
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      global_coords(coords);

  EXPECT_REAL_EQ(global_coords.x, x);
  EXPECT_REAL_EQ(global_coords.z, z);

  // Test that we can create local_coordinates from the returned array
  int ispec = 42;
  specfem::point::local_coordinates<specfem::dimension::type::dim2>
      local_coords(ispec, coords);

  EXPECT_EQ(local_coords.ispec, ispec);
  EXPECT_REAL_EQ(local_coords.xi, x);
  EXPECT_REAL_EQ(local_coords.gamma, z);
}
