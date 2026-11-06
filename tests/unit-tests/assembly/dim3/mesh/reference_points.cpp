#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/element.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <string>

namespace globe_reference_points_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using PointsType = specfem::assembly::mesh_impl::points<dimension>;

// GlobalSmallMesh is meshed with ellipticity on and topography off, so the
// database carries reference anchors and the reference/final difference is
// purely the ellipticity deformation.
const std::string database_path =
    "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
    "proc000000_specfempp_database.bin";

specfem::assembly::mesh<dimension> build_assembly_mesh(
    const specfem::mesh::globe3d_mesh &mesh,
    const specfem::mesh::control_nodes<dimension>::CoordinatesViewType
        &reference_anchors) {
  specfem::quadrature::gll::gll gll{};
  const specfem::quadrature::quadratures quadrature(gll);
  return { mesh.nspec,
           mesh.control_nodes.ngnod,
           mesh.element_grid.ngllz,
           mesh.element_grid.nglly,
           mesh.element_grid.ngllx,
           mesh.tags,
           mesh.adjacency_graph,
           mesh.control_nodes,
           quadrature,
           reference_anchors };
}

// Maximum |reference - final| over all GLL coordinates of one point set.
type_real max_absolute_coordinate_difference(const PointsType &points) {
  type_real max_diff = 0.0;
  for (int ispec = 0; ispec < points.nspec; ++ispec) {
    for (int iz = 0; iz < points.ngllz; ++iz) {
      for (int iy = 0; iy < points.nglly; ++iy) {
        for (int ix = 0; ix < points.ngllx; ++ix) {
          for (int dim = 0; dim < 3; ++dim) {
            max_diff = std::max(
                max_diff,
                std::abs(points.h_reference_coord(ispec, iz, iy, ix, dim) -
                         points.h_coord(ispec, iz, iy, ix, dim)));
          }
        }
      }
    }
  }
  return max_diff;
}

// Maximum |r(reference) - r(final)| over all GLL points of one point set.
type_real max_radial_difference(const PointsType &points) {
  type_real max_diff = 0.0;
  for (int ispec = 0; ispec < points.nspec; ++ispec) {
    for (int iz = 0; iz < points.ngllz; ++iz) {
      for (int iy = 0; iy < points.nglly; ++iy) {
        for (int ix = 0; ix < points.ngllx; ++ix) {
          const type_real xa = points.h_reference_coord(ispec, iz, iy, ix, 0);
          const type_real ya = points.h_reference_coord(ispec, iz, iy, ix, 1);
          const type_real za = points.h_reference_coord(ispec, iz, iy, ix, 2);
          const type_real xb = points.h_coord(ispec, iz, iy, ix, 0);
          const type_real yb = points.h_coord(ispec, iz, iy, ix, 1);
          const type_real zb = points.h_coord(ispec, iz, iy, ix, 2);
          const type_real ra = std::sqrt(xa * xa + ya * ya + za * za);
          const type_real rb = std::sqrt(xb * xb + yb * yb + zb * zb);
          max_diff = std::max(max_diff, std::abs(ra - rb));
        }
      }
    }
  }
  return max_diff;
}

} // namespace globe_reference_points_test_impl

// One test: building assembly::mesh for the globe fixture dominates the
// run time, so all checks share the two constructions.
TEST(GlobeReferencePoints, SamplesModelOnReferenceGeometry) {
  namespace test_impl = globe_reference_points_test_impl;
  constexpr auto dimension = test_impl::dimension;

  const auto mesh = specfem::io::read_globe_mesh(test_impl::database_path,
                                                 specfem::attenuation::Setup{});
  ASSERT_TRUE(mesh.globe.has_reference_geometry)
      << "GlobalSmallMesh is meshed with ellipticity on and must carry "
         "reference anchors";

  const auto mesh_with_reference =
      test_impl::build_assembly_mesh(mesh, mesh.globe.reference_coordinates);
  const auto mesh_without_reference = test_impl::build_assembly_mesh(mesh, {});

  const auto &final_points =
      static_cast<const test_impl::PointsType &>(mesh_with_reference);

  {
    SCOPED_TRACE("reference coordinates are a fresh field");
    EXPECT_NE(final_points.h_reference_coord.data(),
              final_points.h_coord.data());
    EXPECT_NE(final_points.reference_coord.data(), final_points.coord.data());
    EXPECT_EQ(final_points.h_reference_coord.extent(0),
              final_points.h_coord.extent(0));
  }

  {
    SCOPED_TRACE("ellipticity separates reference and final coordinates");
    const auto max_difference = test_impl::max_radial_difference(final_points);
    // The fixture has no topography, so the difference is the ellipticity
    // deformation: kilometre scale at the surface, bounded by the flattening.
    EXPECT_GT(max_difference, 500.0);
    EXPECT_LT(max_difference, 40000.0);
  }

  {
    SCOPED_TRACE("reference coordinates alias the final ones without anchors");
    const auto &fallback_points =
        static_cast<const test_impl::PointsType &>(mesh_without_reference);
    EXPECT_EQ(fallback_points.h_reference_coord.data(),
              fallback_points.h_coord.data());
    EXPECT_EQ(fallback_points.reference_coord.data(),
              fallback_points.coord.data());
  }

  {
    SCOPED_TRACE("identical anchors reproduce final coordinates to round-off");
    // Feed the final anchors in as reference anchors: same anchors, same shape
    // functions, same contraction, so the reference field must reproduce the
    // final one. Costs a third fixture construction — worth it to exercise the
    // production path (mesh ctor -> points ctor with reference nodes).
    const auto mesh_identical =
        test_impl::build_assembly_mesh(mesh, mesh.control_nodes.coordinates);
    const auto &identical =
        static_cast<const test_impl::PointsType &>(mesh_identical);
    EXPECT_NE(identical.h_reference_coord.data(), identical.h_coord.data());
    const auto max_difference =
        test_impl::max_absolute_coordinate_difference(identical);
    // Any difference is pure round-off (1 m is ~1e-7 relative at Earth radius).
    EXPECT_LE(max_difference, 1.0);
  }
}
