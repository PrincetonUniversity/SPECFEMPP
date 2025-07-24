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
  auto coords = base_source.get_global_coordinates();

  // Verify the returned array has correct values
  EXPECT_REAL_EQ(coords.x, x);
  EXPECT_REAL_EQ(coords.z, z);
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

  auto coords = force_source.get_global_coordinates();

  EXPECT_REAL_EQ(coords.x, 0.0);
  EXPECT_REAL_EQ(coords.z, 0.0);
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

  auto coords = force_source.get_global_coordinates();

  EXPECT_REAL_EQ(coords.x, x);
  EXPECT_REAL_EQ(coords.z, z);
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

  auto coords = force_source.get_global_coordinates();

  EXPECT_REAL_EQ(coords.x, x);
  EXPECT_REAL_EQ(coords.z, z);
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

  auto coords_tensor = moment_tensor_source.get_global_coordinates();
  EXPECT_REAL_EQ(coords_tensor.x, x);
  EXPECT_REAL_EQ(coords_tensor.z, z);

  // Test with external source
  specfem::sources::external<specfem::dimension::type::dim2> external_source(
      x, z,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  const auto global_coords_external = external_source.get_global_coordinates();
  EXPECT_REAL_EQ(global_coords_external.x, x);
  EXPECT_REAL_EQ(global_coords_external.z, z);
}

// Test consistency between get_x()/get_z() and get_coords() through base class
TEST(SOURCES_BASE, GetCoords_ConsistentWithGetters) {
  type_real x = 7.5;
  type_real z = -4.2;

  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      global_coords(x, z);

  specfem::sources::force<specfem::dimension::type::dim2> force_source(
      x, z, 0.0,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      specfem::wavefield::simulation_field::forward);

  // Access through base class reference
  specfem::sources::source<specfem::dimension::type::dim2> &base_source =
      force_source;

  auto source_coords = base_source.get_global_coordinates();

  // Verify consistency with existing getters through base class
  EXPECT_REAL_EQ(global_coords.x, source_coords.x);
  EXPECT_REAL_EQ(global_coords.z, source_coords.z);
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

  auto coords = force_source.get_global_coordinates();

  EXPECT_REAL_EQ(coords.x, x);
  EXPECT_REAL_EQ(coords.z, z);

  // Test that we can create local_coordinates from the returned array
  int ispec = 42;
  specfem::point::local_coordinates<specfem::dimension::type::dim2>
      local_coords(ispec, coords.x, coords.z);

  force_source.set_local_coordinates(local_coords);

  EXPECT_EQ(force_source.get_local_coordinates().ispec, ispec);
  EXPECT_REAL_EQ(force_source.get_local_coordinates().xi, x);
  EXPECT_REAL_EQ(force_source.get_local_coordinates().gamma, z);
}
