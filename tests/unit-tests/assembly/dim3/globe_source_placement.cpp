#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/assembly/coordinate_conversion.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <variant>

namespace globe_source_placement_test_impl {

using specfem::element::dimension_tag;
using PointsType = specfem::assembly::mesh_impl::points<dimension_tag::dim3>;

// Spherical (ELLIPTICITY=.false.) single-chunk 1D PREM fixture, centred at
// latitude 40, longitude 10. Placement resolves through the perfect-sphere
// geographic -> geocentric path.
const std::string database_path =
    "data/dim3_globe/single_chunk_1D/"
    "DATABASES_MPI/proc000000_specfempp_database.bin";

// Build an assembly mesh carrying the globe projection (r_planet, ellipticity
// off), so source placement resolves on the sphere. The plain component
// constructor leaves the projection as pass-through Cartesian, so it is set
// explicitly from the globe mesh here.
specfem::assembly::mesh<dimension_tag::dim3>
build_globe_assembly_mesh(const specfem::mesh::globe3d_mesh &mesh) {
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  specfem::assembly::mesh<dimension_tag::dim3> assembled{
    mesh.nspec,
    mesh.control_nodes.ngnod,
    mesh.element_grid.ngllz,
    mesh.element_grid.nglly,
    mesh.element_grid.ngllx,
    mesh.tags,
    mesh.adjacency_graph,
    mesh.control_nodes,
    quadrature,
    {}
  };
  assembled.projection =
      specfem::assembly::mesh_impl::coordinate_projection(mesh);
  return assembled;
}

double planet_radius(const specfem::assembly::mesh<dimension_tag::dim3> &mesh) {
  return std::get<specfem::assembly::mesh_impl::globe_projection>(
             mesh.projection.value)
      .r_planet;
}

specfem::assembly::mesh<dimension_tag::dim3> load() {
  return build_globe_assembly_mesh(specfem::io::read_globe_mesh(
      database_path, specfem::attenuation::Setup{}));
}

} // namespace globe_source_placement_test_impl

// A source at a known latitude/longitude/depth inside the chunk lands in a
// valid element with reference coordinates inside [-1, 1]. The point is offset
// from the chunk centre and set a few tens of km deep so it falls in an element
// interior, not on a shared face/corner (the exact centre with an even NEX sits
// on an element boundary). The bound matches locate_point's contract, which
// accepts/clips reference coordinates to +/-1.01 at element boundaries.
TEST(GlobeSourcePlacement, KnownLatLonLandsInElement) {
  namespace impl = globe_source_placement_test_impl;
  const auto mesh = impl::load();

  const specfem::coordinate_systems::geographic_coordinates geographic(
      15.0, 35.0, 100000.0); // lon, lat, depth [m]
  const auto global = specfem::assembly::to_cartesian(geographic, mesh);

  const auto lcoord =
      specfem::algorithms::locate_point_impl::locate_point(global, mesh);

  EXPECT_GE(lcoord.ispec, 0);
  EXPECT_LE(std::abs(lcoord.xi), 1.0 + 1e-3);
  EXPECT_LE(std::abs(lcoord.eta), 1.0 + 1e-3);
  EXPECT_LE(std::abs(lcoord.gamma), 1.0 + 1e-3);
}

// Depth is measured from the (spherical) surface: r = r_planet - depth.
TEST(GlobeSourcePlacement, DepthResolvesToRadius) {
  namespace impl = globe_source_placement_test_impl;
  const auto mesh = impl::load();

  const double r_planet = impl::planet_radius(mesh);
  const double depth = 50000.0; // 50 km
  const specfem::coordinate_systems::geographic_coordinates geographic(
      10.0, 40.0, depth);
  const auto global = specfem::assembly::to_cartesian(geographic, mesh);

  // Distance from the planet centre (origin).
  EXPECT_NEAR(specfem::point::distance(
                  global,
                  specfem::point::global_coordinates<impl::dimension_tag::dim3>{
                      0.0, 0.0, 0.0 }),
              r_planet - depth, 1.0); // metre tolerance
}

// Round-trip: an element's centre GLL point maps back to its own element with
// reference coordinates at the centre (0, 0, 0). Recovery is limited by
// type_real precision (float in the default build), not the 1e-8 of a double
// build, so the tolerance is set accordingly.
TEST(GlobeSourcePlacement, GllCenterRoundTrip) {
  namespace impl = globe_source_placement_test_impl;
  using specfem::element::dimension_tag;
  const auto mesh = impl::load();
  const auto &points = static_cast<const impl::PointsType &>(mesh);

  ASSERT_EQ(points.ngllx % 2, 1)
      << "expects an odd GLL count so the centre node is at reference 0";
  const int mid_x = points.ngllx / 2;
  const int mid_y = points.nglly / 2;
  const int mid_z = points.ngllz / 2;

  for (int ispec : { 0, points.nspec / 2, points.nspec - 1 }) {
    const specfem::point::global_coordinates<dimension_tag::dim3> global = {
      points.h_coord(ispec, mid_z, mid_y, mid_x, 0),
      points.h_coord(ispec, mid_z, mid_y, mid_x, 1),
      points.h_coord(ispec, mid_z, mid_y, mid_x, 2)
    };
    const auto lcoord =
        specfem::algorithms::locate_point_impl::locate_point(global, mesh);
    SCOPED_TRACE("ispec=" + std::to_string(ispec));
    EXPECT_EQ(lcoord.ispec, ispec);
    EXPECT_NEAR(lcoord.xi, 0.0, 1e-4);
    EXPECT_NEAR(lcoord.eta, 0.0, 1e-4);
    EXPECT_NEAR(lcoord.gamma, 0.0, 1e-4);
  }
}

// A point far above the surface, outside every element, throws rather than
// clamping to an edge element.
TEST(GlobeSourcePlacement, OutsideMeshThrows) {
  namespace impl = globe_source_placement_test_impl;
  using specfem::element::dimension_tag;
  const auto mesh = impl::load();

  const double r_planet = impl::planet_radius(mesh);
  const specfem::point::global_coordinates<dimension_tag::dim3> outside = {
    static_cast<type_real>(2.0 * r_planet), 0.0, 0.0
  };
  EXPECT_THROW(
      (void)specfem::algorithms::locate_point_impl::locate_point(outside, mesh),
      std::runtime_error);
}
