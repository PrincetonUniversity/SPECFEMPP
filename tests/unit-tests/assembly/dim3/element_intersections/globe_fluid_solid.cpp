#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"
#include "specfem/element_coupling.hpp"
#include "specfem/io.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <boost/graph/adjacency_list.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
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

  explicit Fixture(const bool spherical = false)
      : mesh(specfem::io::read_globe_mesh(database_path,
                                          specfem::attenuation::Setup{})) {
    // A synthetic spherical variant uses undeformed anchors as its *physical*
    // geometry. The production fixture always keeps its deformed coordinates.
    if (spherical) {
      Kokkos::deep_copy(mesh.control_nodes.coordinates,
                        mesh.globe.reference_coordinates);
    }
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
      // The CMB and ICB are the only fluid-solid interfaces in the globe, so an
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

namespace globe_fluid_solid_test_impl {

using Vector = std::array<double, 3>;

double norm(const Vector &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vector cross(const Vector &a, const Vector &b) {
  return { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0] };
}

/** @brief Integrals over one side of one named interface. */
struct SurfaceIntegral {
  double area = 0;
  double radius_sum = 0;
  int faces = 0;
  Vector normal_integral{};
  Vector boundary_integral{};
  std::set<int> global_indices;
};

/**
 * @brief Check the assembled geometry and physics, independently of J inverse.
 *
 * Both core interfaces are xi/eta faces. Differentiate physical GLL
 * coordinates along those two axes and take their cross product to obtain
 * an independent oriented surface metric. This also catches a constant normal
 * or a duplicated/missing surface Jacobian in the interface container.
 */
template <specfem::element_coupling::interface_tag InterfaceTag>
SurfaceIntegral check_surface(
    const Fixture &fixture,
    const specfem::assembly::conforming_interfaces<dimension> &interfaces,
    const specfem::element::region_tag solid_region) {
  constexpr bool fluid_self =
      InterfaceTag ==
      specfem::element_coupling::interface_tag::acoustic_elastic;
  constexpr auto boundary = specfem::element::boundary_tag::none;
  const auto [self, coupled] = fixture.intersections(InterfaceTag);
  const auto &mesh = fixture.assembly_mesh;
  const int ngll = mesh.element_grid.ngll;
  // Differentiation of SI coordinates amplifies single-precision roundoff.
  constexpr double metric_tolerance =
      std::max(1e-10, 200.0 * std::numeric_limits<type_real>::epsilon());
  SurfaceIntegral result;
  double normal_variation = 0;
  for (int iface = 0; iface < self.N; ++iface) {
    const int solid_ispec =
        fluid_self ? coupled.element_index(iface) : self.element_index(iface);
    const int raw_solid = mesh.h_compute_to_mesh(solid_ispec);
    if (fixture.mesh.globe.element_context.at(raw_solid).region !=
        solid_region) {
      continue;
    }
    SCOPED_TRACE(::testing::Message()
                 << specfem::element_coupling::to_string(InterfaceTag)
                 << " face=" << iface
                 << " ispec=" << self.element_index(iface));
    ++result.faces;
    const auto face = self(iface);
    const bool cmb = solid_region == specfem::element::region_tag::crust_mantle;
    // Outward from self: fluid CMB +r, fluid ICB -r; solid reverses both.
    const double radial_sign = (cmb == fluid_self) ? 1 : -1;
    // Hex27 represents a sphere only approximately. Scale the angular
    // tolerance with this face's size, especially on the coarse ICB mesh.
    Vector corner{};
    double angular_diameter = 0;
    for (const int i : { 0, ngll - 1 }) {
      for (const int j : { 0, ngll - 1 }) {
        const auto index = face(i, j);
        Vector direction{};
        for (int d = 0; d < 3; ++d) {
          direction[d] =
              mesh.h_coord(index.ispec, index.iz, index.iy, index.ix, d);
        }
        const double radius = norm(direction);
        for (auto &component : direction) {
          component /= radius;
        }
        if (i == 0 && j == 0) {
          corner = direction;
        }
        double cosine = 0;
        for (int d = 0; d < 3; ++d) {
          cosine += corner[d] * direction[d];
        }
        angular_diameter = std::max(angular_diameter,
                                    std::acos(std::clamp(cosine, -1.0, 1.0)));
      }
    }
    const double angular_tolerance = 0.005 + 0.1 * angular_diameter;
    Vector first_normal{};
    for (int i = 0; i < ngll; ++i) {
      for (int j = 0; j < ngll; ++j) {
        const auto index = face(i, j);
        const int ispec = index.ispec;
        const int iz = index.iz;
        const int iy = index.iy;
        const int ix = index.ix;
        specfem::point::conforming_interface<dimension, InterfaceTag, boundary>
            point;
        specfem::assembly::load_on_host(index, interfaces, point);
        Vector position{}, tangent_xi{}, tangent_eta{}, normal{};
        for (int d = 0; d < 3; ++d) {
          position[d] = mesh.h_coord(ispec, iz, iy, ix, d);
          normal[d] = point.face_normal(d);
          for (int k = 0; k < ngll; ++k) {
            tangent_xi[d] +=
                mesh.h_hprime(ix, k) * mesh.h_coord(ispec, iz, iy, k, d);
            tangent_eta[d] +=
                mesh.h_hprime(iy, k) * mesh.h_coord(ispec, iz, k, ix, d);
          }
        }
        const auto metric = cross(tangent_xi, tangent_eta);
        const double surface_jacobian = norm(metric);
        const double normal_length = norm(normal);
        const double radius = norm(position);
        EXPECT_GT(surface_jacobian, 0);
        EXPECT_GT(normal_length, 0);
        const double orientation =
            index.face_type == specfem::mesh_entity::dim3::type::top ? 1 : -1;
        const double weight = mesh.h_weights(ix) * mesh.h_weights(iy);
        EXPECT_NEAR(point.face_factor, weight, metric_tolerance);
        double radial_cosine = 0;
        for (int d = 0; d < 3; ++d) {
          EXPECT_NEAR(normal[d] / surface_jacobian,
                      orientation * metric[d] / surface_jacobian,
                      metric_tolerance);
          radial_cosine +=
              radial_sign * normal[d] * position[d] / (normal_length * radius);
          result.normal_integral[d] += weight * normal[d];
          if (i == 0 && j == 0) {
            first_normal[d] = normal[d] / normal_length;
          }
          normal_variation =
              std::max(normal_variation,
                       std::abs(normal[d] / normal_length - first_normal[d]));
        }
        EXPECT_GT(radial_cosine, std::cos(angular_tolerance));
        // Stokes' theorem: integral(n dS) = 1/2 integral(r x dr) on
        // the oriented rim. Interior edges cancel when faces are summed.
        // This closes the single-chunk patch with radial sidewalls to the
        // origin, whose vector area is minus this boundary integral.
        Vector boundary_tangent{};
        for (int d = 0; d < 3; ++d) {
          if (iy == 0) {
            boundary_tangent[d] += mesh.h_weights(ix) * tangent_xi[d];
          }
          if (ix == ngll - 1) {
            boundary_tangent[d] += mesh.h_weights(iy) * tangent_eta[d];
          }
          if (iy == ngll - 1) {
            boundary_tangent[d] -= mesh.h_weights(ix) * tangent_xi[d];
          }
          if (ix == 0) {
            boundary_tangent[d] -= mesh.h_weights(iy) * tangent_eta[d];
          }
        }
        const auto boundary_area = cross(position, boundary_tangent);
        for (int d = 0; d < 3; ++d) {
          result.boundary_integral[d] += orientation * boundary_area[d] / 2;
        }
        result.area += weight * normal_length;
        result.global_indices.insert(mesh.h_index_mapping(ispec, iz, iy, ix));
        if (i == ngll / 2 && j == ngll / 2) {
          result.radius_sum += radius;
        }

        // Drive the real coupling kernels with radial displacement and unit
        // chi_ddot (pressure p = -chi_ddot). Both kernels use self's outward
        // normal, so there is no extra CMB/ICB sign in the implementation.
        if constexpr (fluid_self) {
          specfem::point::displacement<specfem::tags::Tags<
              dimension, specfem::element::medium_tag::elastic, false>>
              u;
          specfem::point::acceleration<specfem::tags::Tags<
              dimension, specfem::element::medium_tag::acoustic, false>>
              a;
          for (int d = 0; d < 3; ++d) {
            u(d) = position[d] / radius;
          }
          specfem::medium_physics::compute_coupling(point, u, a);
          EXPECT_NEAR(a(0) / (weight * surface_jacobian),
                      radial_sign * radial_cosine, metric_tolerance);
        } else {
          specfem::point::acceleration<specfem::tags::Tags<
              dimension, specfem::element::medium_tag::acoustic, false>>
              chi;
          specfem::point::acceleration<specfem::tags::Tags<
              dimension, specfem::element::medium_tag::elastic, false>>
              a;
          chi(0) = 1;
          specfem::medium_physics::compute_coupling(point, chi, a);
          for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(a(d) / (weight * surface_jacobian),
                        orientation * metric[d] / surface_jacobian,
                        metric_tolerance);
          }
        }
      }
    }
  }
  // Relative to surface area: valid for both the CMB and much smaller ICB.
  for (int d = 0; d < 3; ++d) {
    EXPECT_NEAR((result.normal_integral[d] - result.boundary_integral[d]) /
                    result.area,
                0, metric_tolerance);
  }
  EXPECT_GT(result.faces, 0);
  EXPECT_GT(normal_variation, 1e-3) << "normals must vary across curved faces";
  return result;
}

/** @brief Validate the two directions separately on CMB or ICB geometry. */
void check_interface(const specfem::element::region_tag solid_region,
                     const bool spherical) {
  const Fixture fixture(spherical);
  const auto &mesh = fixture.assembly_mesh;
  const int ngll = mesh.element_grid.ngll;
  const specfem::assembly::jacobian_matrix<dimension> jacobian(mesh);
  const specfem::assembly::conforming_interfaces<dimension> interfaces(
      ngll, ngll, ngll, fixture.element_intersections, jacobian, mesh);
  const auto fluid =
      check_surface<specfem::element_coupling::interface_tag::acoustic_elastic>(
          fixture, interfaces, solid_region);
  const auto solid =
      check_surface<specfem::element_coupling::interface_tag::elastic_acoustic>(
          fixture, interfaces, solid_region);
  EXPECT_EQ(fluid.faces, solid.faces);
  EXPECT_NEAR(solid.area / fluid.area, 1, 1e-5);
  for (int d = 0; d < 3; ++d) {
    EXPECT_NEAR((fluid.normal_integral[d] + solid.normal_integral[d]) /
                    fluid.area,
                0, 1e-5);
  }
  std::vector<int> shared;
  std::set_intersection(fluid.global_indices.begin(),
                        fluid.global_indices.end(),
                        solid.global_indices.begin(),
                        solid.global_indices.end(), std::back_inserter(shared));
  EXPECT_TRUE(shared.empty()) << "fluid and solid share global DOFs";
  if (spherical) {
    // This fixture covers precisely one of six 90-degree cubed-sphere chunks.
    // Test its area against the corresponding solid angle, without pretending
    // its open surface has a zero integral of the normal.
    const double radius = fluid.radius_sum / fluid.faces;
    const double expected = 4 * std::numbers::pi * radius * radius / 6;
    EXPECT_NEAR(fluid.area / expected, 1, 1e-3);
    EXPECT_NEAR(solid.area / expected, 1, 1e-3);
  }
}

} // namespace globe_fluid_solid_test_impl

TEST(GlobeFluidSolidInterfaces, CmbGeometryAndCoupling) {
  globe_fluid_solid_test_impl::check_interface(
      specfem::element::region_tag::crust_mantle, false);
}

TEST(GlobeFluidSolidInterfaces, IcbGeometryAndCoupling) {
  globe_fluid_solid_test_impl::check_interface(
      specfem::element::region_tag::inner_core, false);
}

TEST(GlobeFluidSolidInterfaces, SphericalCmbArea) {
  globe_fluid_solid_test_impl::check_interface(
      specfem::element::region_tag::crust_mantle, true);
}

TEST(GlobeFluidSolidInterfaces, SphericalIcbArea) {
  globe_fluid_solid_test_impl::check_interface(
      specfem::element::region_tag::inner_core, true);
}
