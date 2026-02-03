/**
 * @file connections.cpp
 * @brief Unit tests for 2D spectral element connection mapping
 *
 * This file contains comprehensive unit tests for the 2D element connection
 * functionality in the SPECFEM++ spectral element framework. The tests verify
 * that coordinate mappings between coupled mesh entities (edges, corners) work
 * correctly when two quadrilateral elements share a common boundary.
 *
 * The tests cover:
 * - Edge-to-edge coordinate mapping for all possible edge configurations
 * - Corner-to-corner coordinate mapping at element interfaces
 * - Coordinate2D transformation accuracy for different element orientations
 * - Verification that shared boundaries have matching physical coordinates2D
 *
 * @see specfem::connections::connection_mapping<dim2>
 */

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

/**
 * @brief 2D coordinate structure for testing
 *
 * Represents a point in 2D physical space with x and z coordinates2D.
 * Used for validating coordinate transformations between coupled elements.
 */
struct Coordinate2D {
  type_real x; ///< X-coordinate in physical space
  type_real z; ///< Z-coordinate in physical space

  /**
   * @brief Default constructor initializing to origin
   */
  Coordinate2D() : x(0.0), z(0.0) {}

  /**
   * @brief Parameterized constructor
   *
   * @param x_val X-coordinate value
   * @param z_val Z-coordinate value
   */
  Coordinate2D(const type_real x_val, const type_real z_val)
      : x(x_val), z(z_val) {}

  /**
   * @brief Equality operator with floating-point tolerance
   *
   * Compares two coordinates2D for equality using a relative tolerance
   * to handle floating-point arithmetic precision issues.
   *
   * @param other Coordinate2D to compare against
   * @return true if coordinates2D are approximately equal
   */
  bool operator==(const Coordinate2D &other) const {
    return specfem::utilities::is_close(x, other.x) &&
           specfem::utilities::is_close(z, other.z);
  }

  /**
   * @brief Convert coordinate to string representation
   *
   * @return String in format "(x, z)"
   */
  std::string to_string() const {
    std::ostringstream os;
    os << "(" << x << ", " << z << ")";
    return os.str();
  }
};

/**
 * @brief A simple representation of a quadrilateral element with 4 control
 * nodes.
 *
 * The element is defined by its 4 control nodes located at the corners of a
 * quadrilateral. The local node numbering follows the standard convention:
 *
 * @code{.unparsed}
 *        z
 *        |
 *        |
 *        |/____ x
 *
 *   Node numbering:
 *     3--------2
 *     |        |
 *     |        |
 *     |        |
 *     0--------1
 * @endcode
 *
 * The class provides rotation capabilities to test different element
 * orientations and their connections.
 */
struct TestElement2D {
  /**
   * @brief Control node indices defining element connectivity
   *
   * Array of 4 global node indices that define the quadrilateral element
   * geometry. These indices reference a global node coordinate array.
   */
  Kokkos::View<int *, Kokkos::HostSpace> control_nodes;

  /**
   * @brief Default constructor
   *
   * Creates an element with uninitialized control node indices.
   */
  TestElement2D() : control_nodes("control_nodes", 4) {}

  /**
   * @brief Parameterized constructor with node indices
   *
   * @param nodes Array of 4 global node indices defining element connectivity
   */
  TestElement2D(const std::array<int, 4> nodes)
      : control_nodes("control_nodes", 4) {
    for (int i = 0; i < 4; ++i) {
      this->control_nodes(i) = nodes[i];
    }
  }

  /**
   * @brief Rotate element to align specified edges
   *
   * Rotates the element in the x-z plane to align the 'from' edge with the
   * 'to' edge. This is useful for testing connections between elements with
   * different orientations.
   *
   * The rotation is performed by determining the angle needed to align the
   * edges and applying the appropriate 90-degree rotation(s).
   *
   * @param from Source edge type
   * @param to Target edge type
   */
  void rotate(const specfem::mesh_entity::dim2::type &from,
              const specfem::mesh_entity::dim2::type &to) {
    if (from == to)
      return; // No rotation needed

    // Determine number of 90-degree rotations needed
    int num_rotations = 0;

    // Calculate rotations based on edge transitions
    if ((from == specfem::mesh_entity::dim2::type::top &&
         to == specfem::mesh_entity::dim2::type::right) ||
        (from == specfem::mesh_entity::dim2::type::right &&
         to == specfem::mesh_entity::dim2::type::bottom) ||
        (from == specfem::mesh_entity::dim2::type::bottom &&
         to == specfem::mesh_entity::dim2::type::left) ||
        (from == specfem::mesh_entity::dim2::type::left &&
         to == specfem::mesh_entity::dim2::type::top)) {
      num_rotations = 3; // 270 degrees clockwise
    } else if ((from == specfem::mesh_entity::dim2::type::top &&
                to == specfem::mesh_entity::dim2::type::bottom) ||
               (from == specfem::mesh_entity::dim2::type::bottom &&
                to == specfem::mesh_entity::dim2::type::top) ||
               (from == specfem::mesh_entity::dim2::type::left &&
                to == specfem::mesh_entity::dim2::type::right) ||
               (from == specfem::mesh_entity::dim2::type::right &&
                to == specfem::mesh_entity::dim2::type::left)) {
      num_rotations = 2; // 180 degrees
    } else if ((from == specfem::mesh_entity::dim2::type::top &&
                to == specfem::mesh_entity::dim2::type::left) ||
               (from == specfem::mesh_entity::dim2::type::left &&
                to == specfem::mesh_entity::dim2::type::bottom) ||
               (from == specfem::mesh_entity::dim2::type::bottom &&
                to == specfem::mesh_entity::dim2::type::right) ||
               (from == specfem::mesh_entity::dim2::type::right &&
                to == specfem::mesh_entity::dim2::type::top)) {
      num_rotations = 1; // 90 degrees clockwise (or 90 counter-clockwise)
    } else {
      throw std::runtime_error("Invalid edge-to-edge rotation configuration.");
    }

    // Perform rotations
    for (int i = 0; i < num_rotations; ++i) {
      rotate_90();
    }
  }

private:
  /**
   * @brief Rotate the element 90 degrees clockwise in the x-z plane
   *
   * Performs a single 90-degree clockwise rotation by permuting the control
   * node indices according to the rotation transformation:
   * - Node 0 → Node 1
   * - Node 1 → Node 2
   * - Node 2 → Node 3
   * - Node 3 → Node 0
   */
  void rotate_90() {
    Kokkos::View<int *, Kokkos::HostSpace> rotated_nodes("rotated_nodes", 4);
    rotated_nodes[0] = control_nodes[3];
    rotated_nodes[1] = control_nodes[0];
    rotated_nodes[2] = control_nodes[1];
    rotated_nodes[3] = control_nodes[2];

    control_nodes = rotated_nodes;
  }
};

/**
 * @brief Get global node indices for a given mesh entity
 *
 * Extracts the global node indices that define a specific mesh entity
 * (edge or corner) within an element. This is used to identify matching
 * entities between adjacent elements.
 *
 * @param element The test element containing control node indices
 * @param entity The mesh entity type (edge or corner)
 * @return Vector of global node indices defining the entity
 *
 * @throws std::runtime_error if entity is not an edge or corner
 */
std::vector<int> get_nodes(const TestElement2D &element,
                           const specfem::mesh_entity::dim2::type &entity) {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::edges,
                                     entity)) {
    auto nodes = specfem::mesh_entity::nodes_on_orientation(entity);
    return { element.control_nodes(nodes[0]), element.control_nodes(nodes[1]) };
  } else if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                            entity)) {
    auto nodes = specfem::mesh_entity::nodes_on_orientation(entity);
    return { element.control_nodes(nodes[0]) };
  } else {
    throw std::runtime_error("The provided entity is not an edge or corner");
  }
}

/**
 * @brief Global node coordinates2D for testing
 *
 * Defines 8 nodes arranged in a 2×4 grid to create two stacked quadrilateral
 * elements. The layout allows testing element connections with shared edges.
 *
 * Layout:
 * @code
 *  z=2.0:  6 ------- 7
 *          |         |
 *  z=1.0:  4 ------- 5  <- shared edge between elements
 *          |         |
 *  z=0.0:  0 ------- 1
 *
 *         x=0.0    x=1.0
 * @endcode
 */
const static std::array<specfem::connections_test::Coordinate2D, 8>
    coordinates2D = {
      // z = 0.0 plane
      specfem::connections_test::Coordinate2D{ 0.0, 0.0 },
      specfem::connections_test::Coordinate2D{ 1.0, 0.0 },
      // z = 1.0 plane
      specfem::connections_test::Coordinate2D{ 0.0, 1.0 },
      specfem::connections_test::Coordinate2D{ 1.0, 1.0 },
      // z = 2.0 plane
      specfem::connections_test::Coordinate2D{ 0.0, 2.0 },
      specfem::connections_test::Coordinate2D{ 1.0, 2.0 }
    };

} // namespace specfem::connections_test

/**
 * @brief Configuration structure for parameterized connection tests
 *
 * Defines test parameters for validating coordinate mappings between
 * different edge configurations of adjacent elements.
 */
struct ConnectionTest2DConfig {
  specfem::mesh_entity::dim2::type entity1; ///< Edge on first element
  specfem::mesh_entity::dim2::type entity2; ///< Edge on second element
  std::string name;                         ///< Descriptive test name
};

/**
 * @brief Stream output operator for test configuration
 *
 * Enables readable test names in GoogleTest output.
 */
std::ostream &operator<<(std::ostream &os,
                         const ConnectionTest2DConfig &config) {
  os << config.name;
  return os;
}

/**
 * @brief Parameterized test fixture for coupled 2D elements
 *
 * This test fixture validates coordinate mappings between two adjacent
 * quadrilateral spectral elements that share a common edge. The fixture
 * handles element rotation to test all possible edge-to-edge configurations.
 *
 * The test setup creates two elements:
 * - Element 1: nodes [0, 1, 4, 5] forming bottom quadrilateral
 * - Element 2: nodes [4, 5, 6, 7] forming top quadrilateral
 * - Shared edge: nodes [4, 5] (top of element1 = bottom of element2)
 *
 * @see ConnectionTest2DConfig
 */
class CoupledElements2D
    : public ::testing::TestWithParam<ConnectionTest2DConfig> {
protected:
  /**
   * @brief Test setup - rotate elements to align specified edges
   *
   * Rotates the elements so that the specified edges in the test configuration
   * are aligned, allowing validation of coordinate mapping between different
   * edge orientations.
   */
  void SetUp() override {
    const auto &config = GetParam();
    // Rotate elements so that the specified entities align
    element1.rotate(specfem::mesh_entity::dim2::type::top, config.entity1);
    element2.rotate(specfem::mesh_entity::dim2::type::bottom, config.entity2);

    edges_on_edge1 = specfem::mesh_entity::corners_of_edge(config.entity1);
    edges_on_edge2 = specfem::mesh_entity::corners_of_edge(config.entity2);
  }

  /**
   * @brief Test cleanup - rotate elements back to original orientation
   *
   * Restores elements to their original orientation after testing.
   */
  void TearDown() override {
    const auto &config = GetParam();
    element1.rotate(config.entity1, specfem::mesh_entity::dim2::type::top);
    element2.rotate(config.entity2, specfem::mesh_entity::dim2::type::bottom);
  }

public:
  /**
   * @brief Constructor - initialize two coupled elements
   *
   * Creates two quadrilateral elements with a shared edge:
   * - Element 1: bottom element with nodes [0, 1, 4, 5]
   * - Element 2: top element with nodes [4, 5, 6, 7]
   */
  CoupledElements2D() : element1({ 0, 1, 4, 5 }), element2({ 4, 5, 6, 7 }) {}

  ~CoupledElements2D() override = default;

  specfem::connections_test::TestElement2D element1; ///< First element (source)
  specfem::connections_test::TestElement2D element2; ///< Second element
                                                     ///< (target)

  std::list<specfem::mesh_entity::dim2::type> edges_on_edge1; ///< Corners of
                                                              ///< edge on
                                                              ///< element1
  std::list<specfem::mesh_entity::dim2::type> edges_on_edge2; ///< Corners of
                                                              ///< edge on
                                                              ///< element2
};

/**
 * @brief Compute physical coordinates2D for all GLL points in a 2D element
 *
 * Uses isoparametric transformation with bilinear shape functions to compute
 * the physical coordinates2D of all Gauss-Lobatto-Legendre quadrature points
 * within a quadrilateral element.
 *
 * @param element The test element containing control node indices
 * @return 2D Kokkos view of coordinates2D at all GLL points
 */
Kokkos::View<specfem::connections_test::Coordinate2D **, Kokkos::HostSpace>
compute_coordinates2D(const specfem::connections_test::TestElement2D &element) {
  const int ncontrol_nodes = 4;
  const int ngll = 5; // 5 GLL points per direction
  Kokkos::View<specfem::connections_test::Coordinate2D **, Kokkos::HostSpace>
      coords("coords", ngll, ngll);

  const specfem::quadrature::gll::gll quadrature(0.0, 0.0, ngll);

  const auto xi = quadrature.get_hxi();

  for (int iz = 0; iz < ngll; ++iz) {
    for (int ix = 0; ix < ngll; ++ix) {
      const type_real xil = xi(ix);
      const type_real zetal = xi(iz);
      const auto shape_function =
          specfem::shape_function::shape_function(xil, zetal, ncontrol_nodes);

      type_real x = 0.0;
      type_real z = 0.0;
      for (int a = 0; a < ncontrol_nodes; ++a) {
        const int global_node = element.control_nodes(a);
        x += shape_function[a] *
             specfem::connections_test::coordinates2D[global_node].x;
        z += shape_function[a] *
             specfem::connections_test::coordinates2D[global_node].z;
      }
      coords(iz, ix) = { x, z };
    }
  }

  return coords;
}

/**
 * @brief Test edge-to-edge coordinate mapping between coupled elements
 *
 * Validates that coordinate mapping works correctly for edges shared between
 * adjacent elements. For each point on the shared edge, the test verifies
 * that the mapped coordinates2D produce the same physical location on both
 * elements.
 *
 * The test:
 * 1. Creates a connection mapping between the two elements
 * 2. Iterates over all GLL points on the shared edge
 * 3. Maps each point from element1's edge to element2's edge
 * 4. Verifies that physical coordinates2D match at corresponding points
 */
TEST_P(CoupledElements2D, EdgeConnections) {
  const auto &config = GetParam();
  // Create connection mapping between the two elements
  specfem::mesh_entity::element<specfem::dimension::type::dim2> mapping(5, 5);

  specfem::connections::connection_mapping<specfem::dimension::type::dim2>
      connection(5, 5, element1.control_nodes, element2.control_nodes);

  const int num_points =
      mapping.number_of_points_on_orientation(config.entity1);
  const auto element_coord1 = compute_coordinates2D(element1);
  const auto element_coord2 = compute_coordinates2D(element2);

  for (int ipoint = 0; ipoint < num_points; ++ipoint) {
    const auto [iz1, ix1] = mapping.map_coordinates(config.entity1, ipoint);
    const auto [iz2, ix2] =
        connection.map_coordinates(config.entity1, config.entity2, iz1, ix1);

    const auto coordinate1 = element_coord1(iz1, ix1);
    const auto coordinate2 = element_coord2(iz2, ix2);

    // Verify that the coordinates2D match on the shared edge
    EXPECT_TRUE(coordinate1 == coordinate2)
        << "Mapped coordinates2D do not match for point index " << ipoint
        << " on entities "
        << specfem::mesh_entity::dim2::to_string(config.entity1) << " and "
        << specfem::mesh_entity::dim2::to_string(config.entity2) << ".\n"
        << "Element 1 coordinate: " << coordinate1.to_string() << " at (" << ix1
        << ", " << iz1 << ")\n"
        << "Element 2 coordinate: " << coordinate2.to_string() << " at (" << ix2
        << ", " << iz2 << ")\n";
  }
}

/**
 * @brief Test corner-to-corner coordinate mapping between coupled elements
 *
 * Validates that corner mappings work correctly at element interfaces. For
 * corners that are shared between elements (where node indices match), the
 * test verifies that the mapped coordinates2D produce the same physical
 * location.
 */
TEST_P(CoupledElements2D, CornerConnections) {
  const auto &config = GetParam();
  // Create connection mapping between the two elements
  specfem::mesh_entity::element<specfem::dimension::type::dim2> mapping(5, 5);

  specfem::connections::connection_mapping<specfem::dimension::type::dim2>
      connection(5, 5, element1.control_nodes, element2.control_nodes);

  const auto element_coord1 = compute_coordinates2D(element1);
  const auto element_coord2 = compute_coordinates2D(element2);

  for (const auto corner1 : edges_on_edge1) {
    for (const auto corner2 : edges_on_edge2) {
      if (specfem::connections_test::get_nodes(element1, corner1) ==
          specfem::connections_test::get_nodes(element2, corner2)) {
        // Get local coordinates2D of the corners
        const auto [iz1, ix1] = mapping.map_coordinates(corner1);
        const auto [iz2, ix2] = connection.map_coordinates(corner1, corner2);

        const auto coordinate1 = element_coord1(iz1, ix1);
        const auto coordinate2 = element_coord2(iz2, ix2);

        // Verify that the coordinates2D match on the shared corner
        EXPECT_TRUE(coordinate1 == coordinate2)
            << "Mapped coordinates2D do not match for corner "
            << specfem::mesh_entity::dim2::to_string(corner1) << " on entities "
            << specfem::mesh_entity::dim2::to_string(config.entity1) << " and "
            << specfem::mesh_entity::dim2::to_string(config.entity2) << ".\n"
            << "Element 1 coordinate: " << coordinate1.to_string() << " at ("
            << ix1 << ", " << iz1 << ")\n"
            << "Element 2 coordinate: " << coordinate2.to_string() << " at ("
            << ix2 << ", " << iz2 << ")\n";
      }
    }
  }
}

/**
 * @brief Parameterized test instantiation for all edge-to-edge configurations
 *
 * Tests all possible combinations of edge alignments between two coupled
 * elements, including:
 * - Same orientation (top-bottom, left-right, etc.)
 * - Different orientations (top-left, bottom-right, etc.)
 *
 * This comprehensive test suite ensures the connection mapping works correctly
 * for any possible element adjacency in a 2D mesh.
 */
INSTANTIATE_TEST_SUITE_P(
    ConnectionTest2D, CoupledElements2D,
    ::testing::Values(
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::top,
                                specfem::mesh_entity::dim2::type::bottom,
                                "Top-Bottom" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::top,
                                specfem::mesh_entity::dim2::type::left,
                                "Top-Left" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::top,
                                specfem::mesh_entity::dim2::type::right,
                                "Top-Right" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::top,
                                specfem::mesh_entity::dim2::type::top,
                                "Top-Top" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::bottom,
                                specfem::mesh_entity::dim2::type::top,
                                "Bottom-Top" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::bottom,
                                specfem::mesh_entity::dim2::type::left,
                                "Bottom-Left" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::bottom,
                                specfem::mesh_entity::dim2::type::right,
                                "Bottom-Right" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::bottom,
                                specfem::mesh_entity::dim2::type::bottom,
                                "Bottom-Bottom" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::left,
                                specfem::mesh_entity::dim2::type::top,
                                "Left-Top" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::left,
                                specfem::mesh_entity::dim2::type::bottom,
                                "Left-Bottom" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::left,
                                specfem::mesh_entity::dim2::type::left,
                                "Left-Left" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::left,
                                specfem::mesh_entity::dim2::type::right,
                                "Left-Right" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::right,
                                specfem::mesh_entity::dim2::type::top,
                                "Right-Top" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::right,
                                specfem::mesh_entity::dim2::type::bottom,
                                "Right-Bottom" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::right,
                                specfem::mesh_entity::dim2::type::left,
                                "Right-Left" },
        ConnectionTest2DConfig{ specfem::mesh_entity::dim2::type::right,
                                specfem::mesh_entity::dim2::type::right,
                                "Right-Right" }));
