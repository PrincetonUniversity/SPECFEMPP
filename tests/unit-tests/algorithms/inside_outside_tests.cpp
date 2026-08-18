#include "specfem/algorithms/inside_outside.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <gtest/gtest.h>

// 2D reference-element predicates. outside() is deliberately not !inside():
// a located point in the tolerance band is neither. An unlocated point
// (ispec < 0) counts as outside.
TEST(AlgorithmsInsideOutside, LocalCoordinates2D) {
  using local2d =
      specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>;
  const type_real tolerance = 1.001;

  // Strictly inside
  local2d in(3, 0.5, -0.3);
  EXPECT_TRUE(specfem::algorithms::inside(in));
  EXPECT_FALSE(specfem::algorithms::outside(in, tolerance));

  // On the boundary (|coord| == 1)
  local2d edge(3, 1.0, -1.0);
  EXPECT_TRUE(specfem::algorithms::inside(edge));
  EXPECT_FALSE(specfem::algorithms::outside(edge, tolerance));

  // In the tolerance band (1 < |coord| < tolerance): neither inside nor
  // outside
  local2d band(3, 1.0005, 0.0);
  EXPECT_FALSE(specfem::algorithms::inside(band));
  EXPECT_FALSE(specfem::algorithms::outside(band, tolerance));

  // Beyond the tolerance
  local2d out(3, 1.05, 0.0);
  EXPECT_FALSE(specfem::algorithms::inside(out));
  EXPECT_TRUE(specfem::algorithms::outside(out, tolerance));

  // Unlocated point (ispec < 0): not inside; counts as outside even with
  // in-range coordinates.
  local2d unlocated(-1, 0.5, -0.3);
  EXPECT_FALSE(specfem::algorithms::inside(unlocated));
  EXPECT_TRUE(specfem::algorithms::outside(unlocated, tolerance));
}

// 3D reference-element predicates.
TEST(AlgorithmsInsideOutside, LocalCoordinates3D) {
  using local3d =
      specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>;
  const type_real tolerance = 1.001;

  // Strictly inside
  local3d in(7, 0.5, -0.3, 0.1);
  EXPECT_TRUE(specfem::algorithms::inside(in));
  EXPECT_FALSE(specfem::algorithms::outside(in, tolerance));

  // On the boundary (|coord| == 1)
  local3d edge(7, 1.0, -1.0, 1.0);
  EXPECT_TRUE(specfem::algorithms::inside(edge));
  EXPECT_FALSE(specfem::algorithms::outside(edge, tolerance));

  // In the tolerance band (1 < |eta| < tolerance): neither inside nor outside
  local3d band(7, 0.0, 1.0005, 0.0);
  EXPECT_FALSE(specfem::algorithms::inside(band));
  EXPECT_FALSE(specfem::algorithms::outside(band, tolerance));

  // Beyond the tolerance on gamma
  local3d out(7, 0.0, 0.0, -1.05);
  EXPECT_FALSE(specfem::algorithms::inside(out));
  EXPECT_TRUE(specfem::algorithms::outside(out, tolerance));

  // Unlocated point (ispec < 0): not inside; counts as outside even with
  // in-range coordinates.
  local3d unlocated(-1, 0.5, -0.3, 0.1);
  EXPECT_FALSE(specfem::algorithms::inside(unlocated));
  EXPECT_TRUE(specfem::algorithms::outside(unlocated, tolerance));
}
