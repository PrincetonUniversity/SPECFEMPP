#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace globe_fluid_solid_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;

const std::string database_path =
    "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
    "proc000000_specfempp_database.bin";

/** @brief A face identified by its compute-domain element and orientation. */
using FaceKey = std::pair<int, specfem::mesh_entity::dim3::type>;

/** @brief The two coupling directions across a fluid-solid interface. */
constexpr std::array<specfem::element_coupling::interface_tag, 2> directions = {
  specfem::element_coupling::interface_tag::elastic_acoustic,
  specfem::element_coupling::interface_tag::acoustic_elastic
};

/**
 * @brief Globe assembly built only as far as the derived interface set.
 *
 * Deliberately stops short of the full ``assembly<dim3>``: that would pull in
 * the Fortran model evaluator, sources and receivers, none of which this test
 * needs. Mirrors the staging used by the other globe assembly fixtures.
 */
struct Fixture {
  specfem::mesh::globe3d_mesh mesh;
  specfem::assembly::mesh<dimension> assembly_mesh;
  specfem::assembly::element_types<dimension> element_types;
  specfem::assembly::element_intersections<dimension> element_intersections;

  Fixture()
      : mesh(specfem::io::read_globe_mesh(database_path,
                                          specfem::attenuation::Setup{})) {
    specfem::quadrature::gll::gll gll{};
    const specfem::quadrature::quadratures quadrature(gll);
    assembly_mesh = { mesh.nspec,
                      mesh.control_nodes.ngnod,
                      mesh.element_grid.ngllz,
                      mesh.element_grid.nglly,
                      mesh.element_grid.ngllx,
                      mesh.tags,
                      mesh.adjacency_graph,
                      mesh.control_nodes,
                      quadrature,
                      mesh.globe.reference_coordinates };
    element_types = { mesh.nspec, assembly_mesh.element_grid, assembly_mesh,
                      mesh.tags, mesh.globe.element_context };
    element_intersections = { mesh.element_grid.ngllx, mesh.element_grid.nglly,
                              mesh.element_grid.ngllz, assembly_mesh,
                              element_types };
  }

  /** @brief Self/coupled face views for one coupling direction. */
  auto intersections(
      const specfem::element_coupling::interface_tag interface) const {
    return element_intersections.get_intersections_on_host(
        specfem::element_connections::type::weakly_conforming, interface,
        specfem::element::boundary_tag::none,
        specfem::element_coupling::flux_scheme_tag::natural);
  }
};

/** @brief Faces the assembly derived, over both coupling directions. */
std::set<FaceKey> derived_faces(const Fixture &fixture) {
  std::set<FaceKey> faces;
  for (const auto interface : directions) {
    const auto [self, coupled] = fixture.intersections(interface);
    for (int iface = 0; iface < self.N; ++iface) {
      faces.insert({ self.element_index(iface), self.face_types(iface) });
    }
  }
  return faces;
}

/**
 * @brief Faces the mesh promoted to weakly conforming, in compute ordering.
 *
 * These are the medium-contrast faces @c setup_coupled_interfaces marked on the
 * mesh adjacency graph. The reader's @c check_consistency already ties them to
 * the database's CMB and ICB lists, so this is the assembly's input.
 */
std::set<FaceKey> promoted_faces(const Fixture &fixture) {
  std::set<FaceKey> faces;
  const auto &graph = fixture.mesh.adjacency_graph.local_connections();
  for (const auto v : boost::make_iterator_range(boost::vertices(graph))) {
    for (const auto e :
         boost::make_iterator_range(boost::out_edges(v, graph))) {
      const auto &edge = graph[e];
      if (edge.connection ==
              specfem::element_connections::type::weakly_conforming &&
          specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                         edge.orientation)) {
        faces.insert(
            { fixture.assembly_mesh.h_mesh_to_compute(static_cast<int>(v)),
              edge.orientation });
      }
    }
  }
  return faces;
}

/** @brief Mean radius of the GLL points on one face of one element. */
type_real face_radius(const Fixture &fixture, const int ispec,
                      const specfem::mesh_entity::dim3::type face) {
  const auto &grid = fixture.assembly_mesh.element_grid;
  const int ngll = grid.ngll;
  type_real total = 0.0;
  for (int ipoint = 0; ipoint < ngll; ++ipoint) {
    for (int jpoint = 0; jpoint < ngll; ++jpoint) {
      int iz, iy, ix;
      grid.get_face_coordinates(face, ipoint, jpoint, iz, iy, ix);
      const type_real x = fixture.assembly_mesh.h_coord(ispec, iz, iy, ix, 0);
      const type_real y = fixture.assembly_mesh.h_coord(ispec, iz, iy, ix, 1);
      const type_real z = fixture.assembly_mesh.h_coord(ispec, iz, iy, ix, 2);
      total += std::sqrt(x * x + y * y + z * z);
    }
  }
  return total / static_cast<type_real>(ngll * ngll);
}

/** @brief Render a face set difference for a failure message. */
std::string describe(const std::set<FaceKey> &faces, const std::size_t limit) {
  std::ostringstream message;
  std::size_t shown = 0;
  for (const auto &[ispec, face] : faces) {
    if (shown++ == limit) {
      message << " ... (" << (faces.size() - limit) << " more)";
      break;
    }
    message << " (ispec=" << ispec
            << ", face=" << specfem::mesh_entity::dim3::to_string(face) << ")";
  }
  return message.str();
}

} // namespace globe_fluid_solid_test_impl

// One test: reading the globe database and building the assembly mesh dominates
// the run time, so every check shares a single construction.
TEST(GlobeFluidSolidInterfaces, CollectsPromotedMeshFaces) {
  namespace test_impl = globe_fluid_solid_test_impl;
  using specfem::mesh_entity::dim3::type;

  const test_impl::Fixture fixture;

  int faces_per_direction[test_impl::directions.size()] = { 0, 0 };

  {
    SCOPED_TRACE("both coupling directions are populated and balanced");
    for (std::size_t idirection = 0; idirection < test_impl::directions.size();
         ++idirection) {
      const auto interface = test_impl::directions[idirection];
      const auto [self, coupled] = fixture.intersections(interface);
      SCOPED_TRACE(specfem::element_coupling::to_string(interface));
      // The CMB and ICB are the only medium contrasts in the globe, so an
      // empty set means the weakly-conforming promotion never reached the
      // assembly.
      EXPECT_GT(self.N, 0);
      EXPECT_EQ(self.N, coupled.N);
      faces_per_direction[idirection] = self.N;
    }
    // Every interface face couples in both directions.
    EXPECT_EQ(faces_per_direction[0], faces_per_direction[1]);
  }

  {
    SCOPED_TRACE("every promoted mesh face reaches element_intersections");
    // Agreement with the database's CMB and ICB lists is the reader's job
    // (check_consistency). This checks the assembly's step: the faces it
    // collects are exactly the faces the mesh promoted.
    const auto derived = test_impl::derived_faces(fixture);
    const auto expected = test_impl::promoted_faces(fixture);

    std::set<test_impl::FaceKey> missing;
    std::set<test_impl::FaceKey> unexpected;
    std::set_difference(expected.begin(), expected.end(), derived.begin(),
                        derived.end(), std::inserter(missing, missing.begin()));
    std::set_difference(derived.begin(), derived.end(), expected.begin(),
                        expected.end(),
                        std::inserter(unexpected, unexpected.begin()));

    EXPECT_TRUE(missing.empty())
        << "promoted faces the assembly did not collect:"
        << test_impl::describe(missing, 10);
    EXPECT_TRUE(unexpected.empty())
        << "collected faces the mesh did not promote:"
        << test_impl::describe(unexpected, 10);
    EXPECT_EQ(derived.size(), expected.size());
  }

  {
    SCOPED_TRACE("face orientation agrees with the radial geometry");
    // The database's face codes already match SPECFEM++'s numbering by
    // construction (SPECFEMPP_FACE_BOTTOM = 1, SPECFEMPP_FACE_TOP = 3). What
    // this checks is the geometric half: that the mesher's k = 1 face really is
    // the face SPECFEM++ calls `bottom`. An inversion here would attach the
    // coupling term to the wrong side of every element.
    for (const auto interface : test_impl::directions) {
      const auto [self, coupled] = fixture.intersections(interface);
      for (int iface = 0; iface < self.N; ++iface) {
        const int ispec = self.element_index(iface);
        const auto face = self.face_types(iface);
        // Fluid-solid interfaces in the globe are radial surfaces.
        ASSERT_TRUE(face == type::top || face == type::bottom)
            << "ispec=" << ispec
            << " face=" << specfem::mesh_entity::dim3::to_string(face);

        const auto bottom_radius =
            test_impl::face_radius(fixture, ispec, type::bottom);
        const auto top_radius =
            test_impl::face_radius(fixture, ispec, type::top);
        ASSERT_GT(top_radius, bottom_radius)
            << "element " << ispec << " is not radially oriented";
        if (face == type::bottom) {
          EXPECT_LT(test_impl::face_radius(fixture, ispec, face), top_radius);
        } else {
          EXPECT_GT(test_impl::face_radius(fixture, ispec, face),
                    bottom_radius);
        }
      }
    }
  }
}
