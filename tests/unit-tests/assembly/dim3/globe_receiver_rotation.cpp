#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/receivers.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/receivers.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace globe_receiver_rotation_test_impl {

constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Matrix = std::array<std::array<type_real, 3>, 3>;

// Spherical single-chunk 1D PREM fixture centred at latitude 40, longitude 10.
const std::string database_path =
    "data/dim3_globe/single_chunk_1D/"
    "DATABASES_MPI/proc000000_specfempp_database.bin";

// IU.KONO, inside the chunk (~19.6 degrees from its centre).
constexpr double kono_latitude = 59.6491;
constexpr double kono_longitude = 9.5982;
constexpr double kono_burial = 340.0;

specfem::mesh::globe3d_mesh load_globe_mesh() {
  return specfem::io::read_globe_mesh(database_path,
                                      specfem::attenuation::Setup{});
}

// The component constructor leaves the projection pass-through Cartesian, so
// the globe projection is set explicitly.
specfem::assembly::mesh<dimension>
build_assembly_mesh(const specfem::mesh::globe3d_mesh &mesh) {
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  specfem::assembly::mesh<dimension> assembled{ mesh.nspec,
                                                mesh.control_nodes.ngnod,
                                                mesh.element_grid.ngllz,
                                                mesh.element_grid.nglly,
                                                mesh.element_grid.ngllx,
                                                mesh.tags,
                                                mesh.adjacency_graph,
                                                mesh.control_nodes,
                                                quadrature,
                                                {} };
  assembled.projection =
      specfem::assembly::mesh_impl::coordinate_projection(mesh);
  return assembled;
}

double dot(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> row_of(const Matrix &matrix, const int row) {
  return { static_cast<double>(matrix[row][0]),
           static_cast<double>(matrix[row][1]),
           static_cast<double>(matrix[row][2]) };
}

void expect_vector_near(const std::array<double, 3> &actual,
                        const std::array<double, 3> &expected,
                        const double tolerance) {
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(actual[i], expected[i], tolerance) << "component " << i;
}

} // namespace globe_receiver_rotation_test_impl

// Building the globe assembly mesh dominates the run time, so placement and
// rotation checks share one construction.
TEST(GlobeReceiverRotation, KonoIsPlacedAndRotated) {
  namespace impl = globe_receiver_rotation_test_impl;

  const auto mesh = impl::load_globe_mesh();
  const auto assembly_mesh = impl::build_assembly_mesh(mesh);
  const specfem::assembly::element_types<impl::dimension> element_types{
    mesh.nspec, assembly_mesh.element_grid, assembly_mesh, mesh.tags,
    mesh.globe.element_context
  };

  const auto receiver =
      std::make_shared<specfem::receivers::receiver<impl::dimension>>(
          "IU", "KONO",
          std::make_unique<specfem::coordinate_systems::geographic_coordinates>(
              impl::kono_longitude, impl::kono_latitude, impl::kono_burial));

  const specfem::assembly::receivers<impl::dimension> receivers(
      /*max_sig_step=*/10, /*dt=*/0.01, /*t0=*/0.0,
      /*nsteps_between_samples=*/1, { receiver },
      { specfem::enums::wavefield::displacement }, assembly_mesh,
      element_types);

  {
    SCOPED_TRACE("placement");
    EXPECT_EQ(receiver->get_partition_index(), 0);
    EXPECT_GE(receiver->get_location_error(), 0.0);
    EXPECT_LT(receiver->get_location_error(), 0.05);

    const auto lcoord = specfem::algorithms::locate_point_impl::locate_point(
        receiver->get_global_coordinates(), assembly_mesh);
    EXPECT_GE(lcoord.ispec, 0);
    EXPECT_LE(std::abs(lcoord.xi), 1.001);
    EXPECT_LE(std::abs(lcoord.eta), 1.001);
    EXPECT_LE(std::abs(lcoord.gamma), 1.001);
  }

  const impl::Matrix rotation = receivers.get_rotation_matrix(0);
  const std::array<std::array<double, 3>, 3> rows = {
    impl::row_of(rotation, 0), impl::row_of(rotation, 1),
    impl::row_of(rotation, 2)
  };

  {
    SCOPED_TRACE("orthonormal, right-handed");
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(impl::dot(rows[i], rows[j]), i == j ? 1.0 : 0.0, 1e-6)
            << "rows " << i << ", " << j;

    const auto &east = rows[0];
    const auto &north = rows[1];
    const auto &up = rows[2];
    const std::array<double, 3> east_cross_north = {
      east[1] * north[2] - east[2] * north[1],
      east[2] * north[0] - east[0] * north[2],
      east[0] * north[1] - east[1] * north[0]
    };
    EXPECT_NEAR(impl::dot(east_cross_north, up), 1.0, 1e-6) << "determinant";
  }

  {
    SCOPED_TRACE("Up is radial");
    const auto position = receiver->get_global_coordinates();
    const std::array<double, 3> point = { static_cast<double>(position.x),
                                          static_cast<double>(position.y),
                                          static_cast<double>(position.z) };
    const double radius = std::sqrt(impl::dot(point, point));
    impl::expect_vector_near(
        rows[2], { point[0] / radius, point[1] / radius, point[2] / radius },
        1e-6);
  }

  {
    SCOPED_TRACE("rows match the independently computed local frame");
    // Ellipticity is off in this fixture, so geographic = geocentric. The
    // columns of the Cartesian <- Geocentric rotation are e_theta, e_phi, e_r;
    // East = e_phi, North = -e_theta, Up = e_r.
    constexpr double deg2rad = std::numbers::pi / 180.0;
    const double theta = std::numbers::pi / 2.0 - impl::kono_latitude * deg2rad;
    const double phi = impl::kono_longitude * deg2rad;
    const auto rotation = specfem::coordinate_systems::rotation_matrix<
        specfem::coordinate_systems::CartesianFrame,
        specfem::coordinate_systems::GeocentricFrame>(theta, phi);
    const std::array<double, 3> e_theta = { rotation[0][0], rotation[1][0],
                                            rotation[2][0] };
    const std::array<double, 3> e_phi = { rotation[0][1], rotation[1][1],
                                          rotation[2][1] };
    const std::array<double, 3> e_r = { rotation[0][2], rotation[1][2],
                                        rotation[2][2] };
    impl::expect_vector_near(rows[0], e_phi, 1e-5);
    impl::expect_vector_near(rows[1], { -e_theta[0], -e_theta[1], -e_theta[2] },
                             1e-5);
    impl::expect_vector_near(rows[2], e_r, 1e-5);
  }

  {
    SCOPED_TRACE("round trip through the stored matrix");
    const std::array<double, 3> mesh_vector = { 0.3, -1.2, 2.5 };
    std::array<double, 3> rotated = {};
    for (int i = 0; i < 3; ++i)
      rotated[i] = impl::dot(rows[i], mesh_vector);
    std::array<double, 3> recovered = {};
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i)
        recovered[j] += rows[i][j] * rotated[i];
    impl::expect_vector_near(recovered, mesh_vector, 1e-5);
  }
}
