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
using ReferencePointsType =
    specfem::assembly::mesh_impl::reference_points<dimension>;

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

type_real max_absolute_coordinate_difference(const PointsType &a,
                                             const PointsType &b) {
  type_real max_diff = 0.0;
  for (int ispec = 0; ispec < a.nspec; ++ispec) {
    for (int iz = 0; iz < a.ngllz; ++iz) {
      for (int iy = 0; iy < a.nglly; ++iy) {
        for (int ix = 0; ix < a.ngllx; ++ix) {
          for (int dim = 0; dim < 3; ++dim) {
            max_diff =
                std::max(max_diff, std::abs(a.h_coord(ispec, iz, iy, ix, dim) -
                                            b.h_coord(ispec, iz, iy, ix, dim)));
          }
        }
      }
    }
  }
  return max_diff;
}

type_real max_radial_difference(const PointsType &a, const PointsType &b) {
  type_real max_diff = 0.0;
  for (int ispec = 0; ispec < a.nspec; ++ispec) {
    for (int iz = 0; iz < a.ngllz; ++iz) {
      for (int iy = 0; iy < a.nglly; ++iy) {
        for (int ix = 0; ix < a.ngllx; ++ix) {
          const type_real xa = a.h_coord(ispec, iz, iy, ix, 0);
          const type_real ya = a.h_coord(ispec, iz, iy, ix, 1);
          const type_real za = a.h_coord(ispec, iz, iy, ix, 2);
          const type_real xb = b.h_coord(ispec, iz, iy, ix, 0);
          const type_real yb = b.h_coord(ispec, iz, iy, ix, 1);
          const type_real zb = b.h_coord(ispec, iz, iy, ix, 2);
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
    SCOPED_TRACE("reference set is built and exposed through the accessor");
    ASSERT_TRUE(mesh_with_reference.has_reference_geometry);
    const auto &sampling = mesh_with_reference.model_sampling_coordinates();
    EXPECT_NE(&sampling, &final_points);
    EXPECT_EQ(&sampling, &mesh_with_reference.reference_gll_points);
    EXPECT_EQ(sampling.nspec, final_points.nspec);
    EXPECT_EQ(sampling.ngllz, final_points.ngllz);
    EXPECT_EQ(sampling.nglly, final_points.nglly);
    EXPECT_EQ(sampling.ngllx, final_points.ngllx);
  }

  {
    SCOPED_TRACE("global numbering is shared with the final points");
    const auto &sampling = mesh_with_reference.model_sampling_coordinates();
    EXPECT_EQ(sampling.nglob, final_points.nglob);
    EXPECT_EQ(sampling.h_index_mapping.data(),
              final_points.h_index_mapping.data());
    EXPECT_EQ(sampling.index_mapping.data(), final_points.index_mapping.data());
  }

  {
    SCOPED_TRACE("ellipticity separates reference and final coordinates");
    const auto max_radial_difference = test_impl::max_radial_difference(
        mesh_with_reference.model_sampling_coordinates(), final_points);
    // The fixture has no topography, so the difference is the ellipticity
    // deformation: kilometre scale at the surface, bounded by the flattening.
    EXPECT_GT(max_radial_difference, 500.0);
    EXPECT_LT(max_radial_difference, 40000.0);
  }

  {
    SCOPED_TRACE("accessor falls back to the final points without anchors");
    EXPECT_FALSE(mesh_without_reference.has_reference_geometry);
    const auto &fallback_points =
        static_cast<const test_impl::PointsType &>(mesh_without_reference);
    EXPECT_EQ(&mesh_without_reference.model_sampling_coordinates(),
              &fallback_points);
  }

  {
    SCOPED_TRACE("identical anchors reproduce final coordinates to round-off");
    const test_impl::ReferencePointsType identical(
        final_points,
        static_cast<const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
            dimension> &>(mesh_with_reference),
        static_cast<
            const specfem::assembly::mesh_impl::shape_functions<dimension> &>(
            mesh_with_reference),
        mesh.control_nodes, mesh.control_nodes.coordinates);
    const auto max_difference = test_impl::max_absolute_coordinate_difference(
        identical.reference_gll_points, final_points);
    // Same anchors, same shape functions, same contraction: any difference
    // is pure round-off (1 m is ~1e-7 relative at Earth radius).
    EXPECT_LE(max_difference, 1.0);
  }
}
