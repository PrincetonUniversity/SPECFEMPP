#include "specfem/io/mesh/impl/fortran/dim3/globe_hex27.hpp"
#include "specfem/shape_function.hpp"
#include <array>
#include <gtest/gtest.h>

TEST(GlobeHex27, AnchorOrderingRoundTrip) {
  constexpr std::array<std::array<double, 3>, 27> globe_natural_coordinates = {
    std::array<double, 3>{ -1, -1, -1 },
    { 1, -1, -1 },
    { 1, 1, -1 },
    { -1, 1, -1 },
    { -1, -1, 1 },
    { 1, -1, 1 },
    { 1, 1, 1 },
    { -1, 1, 1 },
    { 0, -1, -1 },
    { 1, 0, -1 },
    { 0, 1, -1 },
    { -1, 0, -1 },
    { -1, -1, 0 },
    { 1, -1, 0 },
    { 1, 1, 0 },
    { -1, 1, 0 },
    { 0, -1, 1 },
    { 1, 0, 1 },
    { 0, 1, 1 },
    { -1, 0, 1 },
    { 0, 0, -1 },
    { 0, -1, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { -1, 0, 0 },
    { 0, 0, 1 },
    { 0, 0, 0 }
  };

  constexpr auto inverse =
      specfem::io::mesh::impl::fortran::dim3::specfem_to_globe_hex27();
  for (int globe_anchor = 0; globe_anchor < 27; ++globe_anchor) {
    const int specfem_anchor = specfem::io::mesh::impl::fortran::dim3::
        globe_to_specfem_hex27[globe_anchor];
    EXPECT_EQ(inverse[specfem_anchor], globe_anchor);

    const auto &point = globe_natural_coordinates[globe_anchor];
    const auto shape = specfem::shape_function::shape_function(
        point[0], point[1], point[2], 27);
    for (int inode = 0; inode < 27; ++inode) {
      EXPECT_DOUBLE_EQ(shape[inode], inode == specfem_anchor ? 1.0 : 0.0)
          << "globe anchor " << globe_anchor << ", SPECFEM++ node " << inode;
    }
  }
}
