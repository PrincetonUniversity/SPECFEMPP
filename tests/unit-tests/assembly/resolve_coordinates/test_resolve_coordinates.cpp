#include "specfem/assembly/resolve_coordinates.hpp"

#include "specfem/coordinate_systems/coordinates/cartesian_2d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_3d.hpp"
#include "specfem/coordinate_systems/coordinates/cartesian_with_depth_3d.hpp"
#include "specfem/coordinate_systems/coordinates/geographic_3d.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

constexpr auto dim2 = specfem::element::dimension_tag::dim2;
constexpr auto dim3 = specfem::element::dimension_tag::dim3;

// Default-constructed mesh — resolve_coordinates doesn't use it for
// the currently implemented coordinate types.
const specfem::assembly::mesh<dim2> mesh2d{};
const specfem::assembly::mesh<dim3> mesh3d{};

} // namespace

TEST(ResolveCoordinates2D, Cartesian2D) {
  specfem::coordinate_systems::cartesian_2d coords(100.0, 200.0);

  auto gc = specfem::assembly::resolve_coordinates(coords, mesh2d);

  EXPECT_FLOAT_EQ(gc.x, 100.0);
  EXPECT_FLOAT_EQ(gc.z, 200.0);
}

TEST(ResolveCoordinates3D, Cartesian3D) {
  specfem::coordinate_systems::cartesian_3d coords(1.0, 2.0, 3.0);

  auto gc = specfem::assembly::resolve_coordinates(coords, mesh3d);

  EXPECT_FLOAT_EQ(gc.x, 1.0);
  EXPECT_FLOAT_EQ(gc.y, 2.0);
  EXPECT_FLOAT_EQ(gc.z, 3.0);
}

TEST(ResolveCoordinates3D, CartesianWithDepth3D) {
  specfem::coordinate_systems::cartesian_with_depth_3d coords(10.0, 20.0, 5.0);

  auto gc = specfem::assembly::resolve_coordinates(coords, mesh3d);

  EXPECT_FLOAT_EQ(gc.x, 10.0);
  EXPECT_FLOAT_EQ(gc.y, 20.0);
  EXPECT_FLOAT_EQ(gc.z, -5.0);
}

TEST(ResolveCoordinates3D, Geographic3DThrows) {
  specfem::coordinate_systems::geographic_3d coords(0.0, 0.0, 0.0);

  EXPECT_THROW(specfem::assembly::resolve_coordinates(coords, mesh3d),
               std::runtime_error);
}
