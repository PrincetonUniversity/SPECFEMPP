
#include "enumerations/connections.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/shape_function.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>
#include <array>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace specfem::connections_test {

enum class Axes2D { X, Y, Z };

struct Coordinate3D {
  type_real x;
  type_real y;
  type_real z;

  Coordinate3D() : x(0.0), y(0.0), z(0.0) {}

  Coordinate3D(const type_real x_val, const type_real y_val,
               const type_real z_val)
      : x(x_val), y(y_val), z(z_val) {}

  bool operator==(const Coordinate3D &other) const {
    return specfem::utilities::is_close(x, other.x) &&
           specfem::utilities::is_close(y, other.y) &&
           specfem::utilities::is_close(z, other.z);
  }

  std::string to_string() const {
    std::ostringstream os;
    os << "(" << x << ", " << y << ", " << z << ")";
    return os.str();
  }
};

/**
 * @brief A simple representation of a hexahedral element with 8 control nodes.
 *
 * The element is defined by its 8 control nodes located at the corners of a
 * hexahedron. The local node numbering follows the standard convention:
 *
 * @code{.unparsed}
 *        z
 *        |  y
 *        | /
 *        |/____ x
 *
 *   Node numbering:
 *        7--------6
 *       /|       /|
 *      / |      / |
 *     4--------5  |
 *     |  |     |  |
 *     |  3-----|--2
 *     | /      | /
 *     |/       |/
 *     0--------1
 * @endcode
 *
 */
struct TestElement3D {
  Kokkos::View<int *, Kokkos::HostSpace> control_nodes;

  TestElement3D() : control_nodes("control_nodes", 8) {}

  TestElement3D(const std::array<int, 8> nodes)
      : control_nodes("control_nodes", 8) {
    for (int i = 0; i < 8; ++i) {
      this->control_nodes(i) = nodes[i];
    }
  }

  void rotate(const specfem::mesh_entity::dim3::type &from,
              const specfem::mesh_entity::dim3::type &to) {
    if (from == to)
      return; // No rotation needed
    // Determine rotation axis and number of 90-degree rotations needed
    std::vector<Axes2D> axes;
    // 3 rotation along x-axis
    if ((from == specfem::mesh_entity::dim3::type::top &&
         to == specfem::mesh_entity::dim3::type::back) ||
        (from == specfem::mesh_entity::dim3::type::back &&
         to == specfem::mesh_entity::dim3::type::bottom) ||
        (from == specfem::mesh_entity::dim3::type::bottom &&
         to == specfem::mesh_entity::dim3::type::front) ||
        (from == specfem::mesh_entity::dim3::type::front &&
         to == specfem::mesh_entity::dim3::type::top)) {
      axes.push_back(Axes2D::X);
    }
    // 3 rotation along y-axis
    else if ((from == specfem::mesh_entity::dim3::type::top &&
              to == specfem::mesh_entity::dim3::type::left) ||
             (from == specfem::mesh_entity::dim3::type::left &&
              to == specfem::mesh_entity::dim3::type::bottom) ||
             (from == specfem::mesh_entity::dim3::type::bottom &&
              to == specfem::mesh_entity::dim3::type::right) ||
             (from == specfem::mesh_entity::dim3::type::right &&
              to == specfem::mesh_entity::dim3::type::top)) {
      axes.push_back(Axes2D::Y);
    }
    // 3 rotation along z-axis
    else if ((from == specfem::mesh_entity::dim3::type::front &&
              to == specfem::mesh_entity::dim3::type::left) ||
             (from == specfem::mesh_entity::dim3::type::left &&
              to == specfem::mesh_entity::dim3::type::back) ||
             (from == specfem::mesh_entity::dim3::type::back &&
              to == specfem::mesh_entity::dim3::type::right) ||
             (from == specfem::mesh_entity::dim3::type::right &&
              to == specfem::mesh_entity::dim3::type::front)) {
      axes.push_back(Axes2D::Z);
    }
    // opposite faces - two rotations along any axis
    else if ((from == specfem::mesh_entity::dim3::type::top &&
              to == specfem::mesh_entity::dim3::type::bottom) ||
             (from == specfem::mesh_entity::dim3::type::bottom &&
              to == specfem::mesh_entity::dim3::type::top)) {
      axes.push_back(Axes2D::X);
      axes.push_back(Axes2D::X);
    } else if ((from == specfem::mesh_entity::dim3::type::left &&
                to == specfem::mesh_entity::dim3::type::right) ||
               (from == specfem::mesh_entity::dim3::type::right &&
                to == specfem::mesh_entity::dim3::type::left)) {
      axes.push_back(Axes2D::Y);
      axes.push_back(Axes2D::Y);
    } else if ((from == specfem::mesh_entity::dim3::type::front &&
                to == specfem::mesh_entity::dim3::type::back) ||
               (from == specfem::mesh_entity::dim3::type::back &&
                to == specfem::mesh_entity::dim3::type::front)) {
      axes.push_back(Axes2D::Z);
      axes.push_back(Axes2D::Z);
    }
    // 3 rotations along x-axis
    else if ((from == specfem::mesh_entity::dim3::type::back &&
              to == specfem::mesh_entity::dim3::type::top) ||
             (from == specfem::mesh_entity::dim3::type::bottom &&
              to == specfem::mesh_entity::dim3::type::back) ||
             (from == specfem::mesh_entity::dim3::type::front &&
              to == specfem::mesh_entity::dim3::type::bottom) ||
             (from == specfem::mesh_entity::dim3::type::top &&
              to == specfem::mesh_entity::dim3::type::front)) {
      axes.push_back(Axes2D::X);
      axes.push_back(Axes2D::X);
      axes.push_back(Axes2D::X);
    }
    // 3 rotations along y-axis
    else if ((from == specfem::mesh_entity::dim3::type::left &&
              to == specfem::mesh_entity::dim3::type::top) ||
             (from == specfem::mesh_entity::dim3::type::bottom &&
              to == specfem::mesh_entity::dim3::type::left) ||
             (from == specfem::mesh_entity::dim3::type::right &&
              to == specfem::mesh_entity::dim3::type::bottom) ||
             (from == specfem::mesh_entity::dim3::type::top &&
              to == specfem::mesh_entity::dim3::type::right)) {
      axes.push_back(Axes2D::Y);
      axes.push_back(Axes2D::Y);
      axes.push_back(Axes2D::Y);
    }
    // 3 rotations along z-axis
    else if ((from == specfem::mesh_entity::dim3::type::left &&
              to == specfem::mesh_entity::dim3::type::front) ||
             (from == specfem::mesh_entity::dim3::type::back &&
              to == specfem::mesh_entity::dim3::type::left) ||
             (from == specfem::mesh_entity::dim3::type::right &&
              to == specfem::mesh_entity::dim3::type::back) ||
             (from == specfem::mesh_entity::dim3::type::front &&
              to == specfem::mesh_entity::dim3::type::right)) {
      axes.push_back(Axes2D::Z);
      axes.push_back(Axes2D::Z);
      axes.push_back(Axes2D::Z);
    } else {
      throw std::runtime_error("Invalid rotation from " +
                               specfem::mesh_entity::dim3::to_string(from) +
                               " to " +
                               specfem::mesh_entity::dim3::to_string(to) + ".");
    }

    // Perform the rotations in sequence
    for (const auto &axis : axes) {
      rotate(axis);
    }
  }

private:
  void rotate(const Axes2D axis) {
    switch (axis) {
    case Axes2D::X:
      rotate_x();
      break;
    case Axes2D::Y:
      rotate_y();
      break;
    case Axes2D::Z:
      rotate_z();
      break;
    default:
      throw std::runtime_error("Invalid rotation axis.");
    }
  }

  // Rotate the element 90 degrees around the Z-axis
  void rotate_z() {
    Kokkos::View<int *, Kokkos::HostSpace> rotated_nodes("rotated_nodes", 8);
    rotated_nodes[3] = control_nodes[0];
    rotated_nodes[0] = control_nodes[1];
    rotated_nodes[1] = control_nodes[2];
    rotated_nodes[2] = control_nodes[3];
    rotated_nodes[7] = control_nodes[4];
    rotated_nodes[4] = control_nodes[5];
    rotated_nodes[5] = control_nodes[6];
    rotated_nodes[6] = control_nodes[7];

    control_nodes = rotated_nodes;
    return;
  }

  // Rotate the element 90 degrees around the Y-axis
  void rotate_y() {
    Kokkos::View<int *, Kokkos::HostSpace> rotated_nodes("control_nodes", 8);
    rotated_nodes[1] = control_nodes[0];
    rotated_nodes[5] = control_nodes[1];
    rotated_nodes[6] = control_nodes[2];
    rotated_nodes[2] = control_nodes[3];
    rotated_nodes[0] = control_nodes[4];
    rotated_nodes[4] = control_nodes[5];
    rotated_nodes[7] = control_nodes[6];
    rotated_nodes[3] = control_nodes[7];

    control_nodes = rotated_nodes;
    return;
  }

  // Rotate the element 90 degrees around the X-axis
  void rotate_x() {
    Kokkos::View<int *, Kokkos::HostSpace> rotated_nodes("control_nodes", 8);
    rotated_nodes[4] = control_nodes[0];
    rotated_nodes[5] = control_nodes[1];
    rotated_nodes[1] = control_nodes[2];
    rotated_nodes[0] = control_nodes[3];
    rotated_nodes[7] = control_nodes[4];
    rotated_nodes[6] = control_nodes[5];
    rotated_nodes[2] = control_nodes[6];
    rotated_nodes[3] = control_nodes[7];

    control_nodes = rotated_nodes;
    return;
  }
};

std::vector<int> get_nodes(const TestElement3D &element,
                           const specfem::mesh_entity::dim3::type &entity) {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::edges,
                                     entity)) {
    auto nodes = specfem::mesh_entity::nodes_on_orientation(entity);
    return { element.control_nodes(nodes[0]), element.control_nodes(nodes[1]) };
  } else if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                            entity)) {
    auto nodes = specfem::mesh_entity::nodes_on_orientation(entity);
    return { element.control_nodes(nodes[0]) };
  } else {
    throw std::runtime_error("The provided entity is not an edge or corner");
    ;
  }
}

// Define 12 nodes
const static std::array<specfem::connections_test::Coordinate3D, 12>
    coordinates3D = {
      // z = 0.0 plane
      specfem::connections_test::Coordinate3D{ 0.0, 0.0, 0.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 0.0, 0.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 1.0, 0.0 },
      specfem::connections_test::Coordinate3D{ 0.0, 1.0, 0.0 },
      // z = 1.0 plane
      specfem::connections_test::Coordinate3D{ 0.0, 0.0, 1.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 0.0, 1.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 1.0, 1.0 },
      specfem::connections_test::Coordinate3D{ 0.0, 1.0, 1.0 },
      // z = 2.0 plane
      specfem::connections_test::Coordinate3D{ 0.0, 0.0, 2.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 0.0, 2.0 },
      specfem::connections_test::Coordinate3D{ 1.0, 1.0, 2.0 },
      specfem::connections_test::Coordinate3D{ 0.0, 1.0, 2.0 },
    };

} // namespace specfem::connections_test

struct ConnectionTest3DConfig {
  specfem::mesh_entity::dim3::type entity1;
  specfem::mesh_entity::dim3::type entity2;
  std::string name;
};

std::ostream &operator<<(std::ostream &os,
                         const ConnectionTest3DConfig &config) {
  os << config.name;
  return os;
}

class CoupledElements3D
    : public ::testing::TestWithParam<ConnectionTest3DConfig> {
protected:
  void SetUp() override {
    const auto &config = GetParam();
    // We rotate the elements so that the specified entities align
    element1.rotate(specfem::mesh_entity::dim3::type::top, config.entity1);
    element2.rotate(specfem::mesh_entity::dim3::type::bottom, config.entity2);

    edges_on_face1 = specfem::mesh_entity::edges_of_face(config.entity1);
    edges_on_face2 = specfem::mesh_entity::edges_of_face(config.entity2);
  }

  void TearDown() override {
    // Rotate back to original orientation for potential reuse
    const auto &config = GetParam();
    element1.rotate(config.entity1, specfem::mesh_entity::dim3::type::top);
    element2.rotate(config.entity2, specfem::mesh_entity::dim3::type::bottom);
  }

public:
  // Define two elements s.t. element1's top face coincides with element2's
  // bottom face
  CoupledElements3D()
      : element1({ 0, 1, 2, 3, 4, 5, 6, 7 }),
        element2({ 4, 5, 6, 7, 8, 9, 10, 11 }) {}

  ~CoupledElements3D() override = default;

  specfem::connections_test::TestElement3D element1;
  specfem::connections_test::TestElement3D element2;

  std::array<specfem::mesh_entity::dim3::type, 4> edges_on_face1;
  std::array<specfem::mesh_entity::dim3::type, 4> edges_on_face2;
};

Kokkos::View<specfem::connections_test::Coordinate3D ***, Kokkos::HostSpace>
compute_coordinates3D(const specfem::connections_test::TestElement3D &element) {
  const int ncontrol_nodes = 8;
  const int ngll = 5; // 5 GLL points per direction
  Kokkos::View<specfem::connections_test::Coordinate3D ***, Kokkos::HostSpace>
      coords("coords", ngll, ngll, ngll);

  const specfem::quadrature::gll::gll quadrature(0.0, 0.0, ngll);

  const auto xi = quadrature.get_hxi();

  for (int iz = 0; iz < ngll; ++iz) {
    for (int iy = 0; iy < ngll; ++iy) {
      for (int ix = 0; ix < ngll; ++ix) {
        const type_real xil = xi(ix);
        const type_real etal = xi(iy);
        const type_real zetal = xi(iz);
        const auto shape_function = specfem::shape_function::shape_function(
            xil, etal, zetal, ncontrol_nodes);

        type_real x = 0.0;
        type_real y = 0.0;
        type_real z = 0.0;
        for (int a = 0; a < ncontrol_nodes; ++a) {
          x +=
              shape_function[a] *
              specfem::connections_test::coordinates3D[element.control_nodes[a]]
                  .x;
          y +=
              shape_function[a] *
              specfem::connections_test::coordinates3D[element.control_nodes[a]]
                  .y;
          z +=
              shape_function[a] *
              specfem::connections_test::coordinates3D[element.control_nodes[a]]
                  .z;
        }
        coords(iz, iy, ix) = { x, y, z };
      }
    }
  }

  return coords;
}

TEST_P(CoupledElements3D, FaceConnections) {
  const auto &config = GetParam();
  // Create connection mapping between the two elements
  specfem::mesh_entity::element mapping(5, 5, 5);

  specfem::connections::connection_mapping<specfem::dimension::type::dim3>
      connection(5, 5, 5, element1.control_nodes, element2.control_nodes);

  const int num_points =
      mapping.number_of_points_on_orientation(config.entity1);
  const auto element_coord1 = compute_coordinates3D(element1);
  const auto element_coord2 = compute_coordinates3D(element2);
  for (int ipoint = 0; ipoint < num_points; ++ipoint) {
    const auto [iz1, iy1, ix1] =
        mapping.map_coordinates(config.entity1, ipoint);
    const auto [iz2, iy2, ix2] = connection.map_coordinates(
        config.entity1, config.entity2, iz1, iy1, ix1);

    const auto coordinate1 = element_coord1(iz1, iy1, ix1);
    const auto coordinate2 = element_coord2(iz2, iy2, ix2);
    // Verify that the coordinates3D match on the shared face
    EXPECT_TRUE(coordinate1 == coordinate2)
        << "Mapped coordinates3D do not match for point index " << ipoint
        << " on entities "
        << specfem::mesh_entity::dim3::to_string(config.entity1) << " and "
        << specfem::mesh_entity::dim3::to_string(config.entity2) << ".\n"
        << "Element 1 coordinate: " << coordinate1.to_string() << " at (" << ix1
        << ", " << iy1 << ", " << iz1 << ")\n"
        << "Element 2 coordinate: " << coordinate2.to_string() << " at (" << ix2
        << ", " << iy2 << ", " << iz2 << ")\n";
  }
}

TEST_P(CoupledElements3D, EdgeConnections) {
  const auto &config = GetParam();
  // Create connection mapping between the two elements
  specfem::mesh_entity::element mapping(5, 5, 5);

  specfem::connections::connection_mapping<specfem::dimension::type::dim3>
      connection(5, 5, 5, element1.control_nodes, element2.control_nodes);

  // Test edge connections

  const auto element_coord1 = compute_coordinates3D(element1);
  const auto element_coord2 = compute_coordinates3D(element2);
  for (const auto edge1 : edges_on_face1) {
    for (const auto edge2 : edges_on_face2) {
      if (specfem::connections_test::get_nodes(element1, edge1) ==
          specfem::connections_test::get_nodes(element2, edge2)) {
        const int num_points = mapping.number_of_points_on_orientation(edge1);
        for (int ipoint = 0; ipoint < num_points; ++ipoint) {
          const auto [iz1, iy1, ix1] = mapping.map_coordinates(edge1, ipoint);
          const auto [iz2, iy2, ix2] =
              connection.map_coordinates(edge1, edge2, iz1, iy1, ix1);

          const auto coordinate1 = element_coord1(iz1, iy1, ix1);
          const auto coordinate2 = element_coord2(iz2, iy2, ix2);
          // Verify that the coordinates3D match on the shared edge
          EXPECT_TRUE(coordinate1 == coordinate2)
              << "Mapped coordinates3D do not match for point index " << ipoint
              << " on edges " << specfem::mesh_entity::dim3::to_string(edge1)
              << " and " << specfem::mesh_entity::dim3::to_string(edge2)
              << ".\n"
              << "Element 1 coordinate: " << coordinate1.to_string() << " at ("
              << ix1 << ", " << iy1 << ", " << iz1 << ")\n"
              << "Element 2 coordinate: " << coordinate2.to_string() << " at ("
              << ix2 << ", " << iy2 << ", " << iz2 << ")\n";
        }
      }
    }
  }
}

TEST_P(CoupledElements3D, NodeConnections) {
  const auto &config = GetParam();
  // Create connection mapping between the two elements
  specfem::mesh_entity::element mapping(5, 5, 5);

  specfem::connections::connection_mapping<specfem::dimension::type::dim3>
      connection(5, 5, 5, element1.control_nodes, element2.control_nodes);

  const auto element_coord1 = compute_coordinates3D(element1);
  const auto element_coord2 = compute_coordinates3D(element2);
  for (const auto node1 :
       specfem::mesh_entity::corners_of_face(config.entity1)) {
    for (const auto node2 :
         specfem::mesh_entity::corners_of_face(config.entity2)) {
      if (specfem::connections_test::get_nodes(element1, node1) ==
          specfem::connections_test::get_nodes(element2, node2)) {
        // Get local coordinates3D of the nodes
        const auto [iz1, iy1, ix1] = mapping.map_coordinates(node1);
        const auto [iz2, iy2, ix2] = connection.map_coordinates(node1, node2);
        const auto coordinate1 = element_coord1(iz1, iy1, ix1);
        const auto coordinate2 = element_coord2(iz2, iy2, ix2);
        // Verify that the coordinates3D match on the shared node
        EXPECT_TRUE(coordinate1 == coordinate2)
            << "Mapped coordinates3D do not match for node "
            << specfem::mesh_entity::dim3::to_string(node1) << " on entities "
            << specfem::mesh_entity::dim3::to_string(config.entity1) << " and "
            << specfem::mesh_entity::dim3::to_string(config.entity2) << ".\n"
            << "Element 1 coordinate: " << coordinate1.to_string() << " at ("
            << ix1 << ", " << iy1 << ", " << iz1 << ")\n"
            << "Element 2 coordinate: " << coordinate2.to_string() << " at ("
            << ix2 << ", " << iy2 << ", " << iz2 << ")\n";
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConnectionTest3D, CoupledElements3D,
    ::testing::Values(
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Top-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::front,
                                "Top-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::back,
                                "Top-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::left,
                                "Top-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::right,
                                "Top-Right" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::top,
                                specfem::mesh_entity::dim3::type::top,
                                "Top-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::top,
                                "Bottom-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::front,
                                "Bottom-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::back,
                                "Bottom-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::left,
                                "Bottom-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::right,
                                "Bottom-Right" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::bottom,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Bottom-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::top,
                                "Front-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Front-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::front,
                                "Front-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::back,
                                "Front-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::left,
                                "Front-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::front,
                                specfem::mesh_entity::dim3::type::right,
                                "Front-Right" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::top,
                                "Back-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Back-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::front,
                                "Back-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::back,
                                "Back-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::left,
                                "Back-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::back,
                                specfem::mesh_entity::dim3::type::right,
                                "Back-Right" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::top,
                                "Left-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Left-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::front,
                                "Left-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::back,
                                "Left-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::left,
                                "Left-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::left,
                                specfem::mesh_entity::dim3::type::right,
                                "Left-Right" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::top,
                                "Right-Top" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::bottom,
                                "Right-Bottom" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::front,
                                "Right-Front" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::back,
                                "Right-Back" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::left,
                                "Right-Left" },
        ConnectionTest3DConfig{ specfem::mesh_entity::dim3::type::right,
                                specfem::mesh_entity::dim3::type::right,
                                "Right-Right" }));
