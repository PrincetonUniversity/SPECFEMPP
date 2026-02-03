
/**
 * @file single_element.cpp
 * @brief Unit tests for 3D spectral element connectivity mapping
 *
 * This file contains comprehensive unit tests for the 3D element connectivity
 * functionality in the SPECFEM++ spectral element framework. The tests verify
 * that coordinate mappings between mesh entities (faces, edges, corners) and
 * their corresponding grid points work correctly for a single hexahedral
 * element.
 *
 * The tests cover:
 * - Face coordinate mapping for all 6 hexahedral faces
 * - Edge coordinate mapping for all 12 hexahedral edges
 * - Corner coordinate mapping for all 8 hexahedral corners
 * - Point count validation for different mesh entity types
 * - Coordinate transformation accuracy using spectral element shape functions
 *
 * @see specfem::mesh_entity::dim3
 */

#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/shape_function.hpp"
#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

namespace specfem::mesh_entity_test {

/**
 * @brief Flexible coordinate representation for spectral element testing
 *
 * This structure provides a coordinate system that supports both exact values
 * and wildcards (All) for flexible coordinate matching in tests. The wildcard
 * functionality is essential for testing mesh entity mappings where certain
 * coordinate components vary while others remain constant.
 *
 * @code
 * // Exact coordinate specification
 * Coordinate corner(1.0, 0.0, 1.0);
 *
 * // Wildcard coordinate for face testing (x=1, y and z vary)
 * Coordinate right_face(1.0, Coordinate::All(), Coordinate::All());
 *
 * // Test coordinate equality with wildcards
 * Coordinate test_point(1.0, 0.5, 0.7);
 * assert(test_point == right_face); // True: x matches, y and z are wildcards
 * @endcode
 */
struct Coordinate3D {

  /**
   * @brief Wildcard type for flexible coordinate matching
   *
   * The All type represents a wildcard that matches any coordinate value.
   * This is used in test scenarios where only specific coordinate components
   * need to be validated while others can vary freely.
   */
  struct All {
    /**
     * @brief Stream output operator for wildcard representation
     * @param os Output stream
     * @return Reference to the output stream
     */
    std::ostream &operator<<(std::ostream &os) const {
      return os << "*"; // Wildcard representation
    }
  };

  /** @brief X-coordinate: either a real value or wildcard */
  std::variant<type_real, All> x = static_cast<type_real>(0.0);

  /** @brief Y-coordinate: either a real value or wildcard */
  std::variant<type_real, All> y = static_cast<type_real>(0.0);

  /** @brief Z-coordinate: either a real value or wildcard */
  std::variant<type_real, All> z = static_cast<type_real>(0.0);

  /**
   * @brief Default constructor
   *
   * Creates a coordinate at the origin (0, 0, 0).
   */
  Coordinate3D() = default;

  /**
   * @brief Generic constructor with type-safe coordinate assignment
   *
   * Constructs a coordinate with flexible type handling, supporting both
   * floating-point values and wildcard (All) types for any coordinate
   * component. This template-based approach ensures type safety while allowing
   * maximum flexibility in coordinate specification.
   *
   * @tparam X Type of x-coordinate (type_real or All)
   * @tparam Y Type of y-coordinate (type_real or All)
   * @tparam Z Type of z-coordinate (type_real or All)
   * @param x_val X-coordinate value or wildcard
   * @param y_val Y-coordinate value or wildcard
   * @param z_val Z-coordinate value or wildcard
   *
   * @code
   * // Various coordinate construction examples
   * Coordinate origin(0.0, 0.0, 0.0);
   * Coordinate corner(1.0, 1.0, 1.0);
   * Coordinate x_face(1.0, Coordinate::All(), Coordinate::All());
   * Coordinate edge(1.0, 0.0, Coordinate::All());
   * @endcode
   */
  template <typename X = type_real, typename Y = type_real,
            typename Z = type_real,
            typename std::enable_if_t<
                std::is_floating_point_v<X> || std::is_same_v<X, All>, int> = 0,
            typename std::enable_if_t<
                std::is_floating_point_v<Y> || std::is_same_v<Y, All>, int> = 0,
            typename std::enable_if_t<
                std::is_floating_point_v<Z> || std::is_same_v<Z, All>, int> = 0>
  Coordinate3D(X x_val, Y y_val, Z z_val)
      : x([&]() -> std::variant<type_real, All> {
          if constexpr (std::is_same_v<X, All>) {
            return All{};
          } else {
            return static_cast<type_real>(x_val);
          }
        }()),
        y([&]() -> std::variant<type_real, All> {
          if constexpr (std::is_same_v<Y, All>) {
            return All{};
          } else {
            return static_cast<type_real>(y_val);
          }
        }()),
        z([&]() -> std::variant<type_real, All> {
          if constexpr (std::is_same_v<Z, All>) {
            return All{};
          } else {
            return static_cast<type_real>(z_val);
          }
        }()) {}

  /**
   * @brief Initializer list constructor for convenient coordinate creation
   *
   * Constructs a coordinate from an initializer list of exactly 3 values
   * representing x, y, and z coordinates respectively.
   *
   * @param list Initializer list containing exactly 3 coordinate values
   * @throws std::runtime_error if the list doesn't contain exactly 3 values
   *
   * @code
   * Coordinate corner = {1.0, 1.0, 1.0};
   * Coordinate origin = {0.0, 0.0, 0.0};
   * @endcode
   */
  Coordinate3D(const std::initializer_list<type_real> &list) {
    if (list.size() != 3) {
      throw std::runtime_error(
          "Coordinate assignment requires exactly 3 values.");
    }
    auto it = list.begin();
    x = *it++;
    y = *it++;
    z = *it;
  }

  /**
   * @brief Equality operator with wildcard support
   *
   * Compares two coordinates for equality, treating wildcard (All) values
   * as matching any coordinate value. This is essential for testing mesh
   * entity mappings where certain coordinates are constrained while others
   * vary.
   *
   * @param other Coordinate to compare against
   * @return true if coordinates are equal (considering wildcards), false
   * otherwise
   *
   * @code
   * Coordinate exact(1.0, 0.5, 0.0);
   * Coordinate pattern(1.0, Coordinate::All(), 0.0);
   * assert(exact == pattern); // True: x and z match, y is wildcard
   * @endcode
   */
  bool operator==(const Coordinate3D &other) const {
    bool x_equal = false;
    if (std::holds_alternative<All>(x) ||
        std::holds_alternative<All>(other.x)) {
      x_equal = true; // Wildcard matches any value
    } else {
      x_equal = (specfem::utilities::is_close(std::get<type_real>(x),
                                              std::get<type_real>(other.x)));
    }

    bool y_equal = false;
    if (std::holds_alternative<All>(y) ||
        std::holds_alternative<All>(other.y)) {
      y_equal = true; // Wildcard matches any value
    } else {
      y_equal = (specfem::utilities::is_close(std::get<type_real>(y),
                                              std::get<type_real>(other.y)));
    }

    bool z_equal = false;
    if (std::holds_alternative<All>(z) ||
        std::holds_alternative<All>(other.z)) {
      z_equal = true; // Wildcard matches any value
    } else {
      z_equal = (specfem::utilities::is_close(std::get<type_real>(z),
                                              std::get<type_real>(other.z)));
    }

    return x_equal && y_equal && z_equal;
  }

  std::string to_string() const {
    std::ostringstream os;
    os << "(";
    if (std::holds_alternative<All>(x)) {
      os << "*";
    } else {
      os << std::get<type_real>(x);
    }
    os << ", ";
    if (std::holds_alternative<All>(y)) {
      os << "*";
    } else {
      os << std::get<type_real>(y);
    }
    os << ", ";
    if (std::holds_alternative<All>(z)) {
      os << "*";
    } else {
      os << std::get<type_real>(z);
    }
    os << ")";
    return os.str();
  }
};

/**
 * @brief 8-node hexahedral spectral element for connection mapping tests
 *
 * This class represents a 3D hexahedral spectral element with 8 control nodes
 * and Gauss-Lobatto-Legendre (GLL) quadrature points. It provides the geometric
 * foundation for testing coordinate mapping between mesh entities and their
 * corresponding grid points in the spectral element method.
 *
 * The element uses isoparametric mapping with trilinear shape functions to
 * transform between the reference element coordinates (\f$\xi, \eta, \zeta \in
 * [-1,1]^3\f$) and physical coordinates. The 8 control nodes define the element
 * geometry:
 *
 * @code
 * Node numbering (standard hexahedral convention):
 *     7----------6
 *    /|         /|
 *   / |        / |
 *  4----------5  |
 *  |  |       |  |
 *  |  3-------|--2
 *  | /        | /
 *  |/         |/
 *  0----------1
 * @endcode
 *
 * @see specfem::shape_function::shape_function
 * @see specfem::quadrature::gll::gll
 */
struct Element8Node {

  /** @brief Dimension tag for 3D spectral elements */
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /** @brief Number of control nodes in hexahedral element */
  constexpr static int ncontrol_nodes = 8;

  /**
   * @brief Gauss-Lobatto-Legendre quadrature object
   *
   * Provides quadrature points and weights for numerical integration
   * over the reference element domain [-1, 1]³.
   */
  specfem::quadrature::gll::gll quadrature;

  /**
   * @brief Number of GLL points in each spatial dimension
   *
   * For cubic elements, this is the same in all three dimensions (ngllx = nglly
   * = ngllz).
   */
  int ngll;

  /**
   * @brief Physical coordinates of the 8 hexahedral control nodes
   *
   * These coordinates define the element geometry in physical space.
   * The nodes are ordered according to standard hexahedral convention
   * with node 0 at the origin and node 6 at the opposite corner.
   */
  std::array<Coordinate3D, ncontrol_nodes> control_node_coords;

  /**
   * @brief Physical coordinates of all GLL quadrature points
   *
   * 3D Kokkos view storing the transformed coordinates of every GLL point
   * within the element. Layout: quadrature_coords(iz, iy, ix) where
   * indices correspond to the ζ, η, ξ directions respectively.
   */
  Kokkos::View<Coordinate3D ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
      quadrature_coords;

  /**
   * @brief Default constructor
   *
   * Creates an uninitialized element. Use the parameterized constructor
   * for proper element setup with quadrature points and control nodes.
   */
  Element8Node() = default;

  /**
   * @brief Parameterized constructor with control node specification
   *
   * Creates a hexahedral spectral element with the specified number of GLL
   * points and control node coordinates. Automatically computes all quadrature
   * point coordinates using isoparametric transformation.
   *
   * @param ngll Number of GLL quadrature points per dimension
   * @param coords Initializer list of 8 control node coordinates in standard
   * order
   *
   * @throws std::runtime_error if coords doesn't contain exactly 8 coordinates
   *
   * @code
   * // Create unit cube element with 5 GLL points per dimension
   * Element8Node unit_cube(5, {
   *     {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
   *     {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}
   * });
   * @endcode
   */
  Element8Node(const int ngll,
               const std::initializer_list<Coordinate3D> &coords)
      : ngll(ngll), quadrature(0.0, 0.0, ngll),
        quadrature_coords("quad_coords", ngll, ngll, ngll),
        control_node_coords() {
    if (coords.size() != ncontrol_nodes) {
      throw std::runtime_error(
          "Element8Node requires exactly 8 control node coordinates.");
    }
    std::copy(coords.begin(), coords.end(), control_node_coords.begin());
    compute_coordinates();
  }

  /**
   * @brief Compute physical coordinates for all GLL quadrature points
   *
   * Uses isoparametric transformation with trilinear hexahedral shape functions
   * to map from reference element coordinates (\f$\xi, \eta, \zeta\f$) to
   * physical coordinates. The transformation is:
   *
   * \f[
   * \mathbf{x}(\xi, \eta, \zeta) = \sum_{a=1}^{8} N_a(\xi, \eta, \zeta)
   * \mathbf{x}_a
   * \f]
   *
   * where \f$N_a\f$ are the trilinear shape functions and \f$\mathbf{x}_a\f$
   * are the control node coordinates.
   *
   * @note This function is automatically called by the constructor and should
   *       be called manually only if control node coordinates are modified.
   */
  void compute_coordinates() {
    const auto &xi = quadrature.get_hxi();

    assert(std::holds_alternative<type_real>(control_node_coords[0].x) &&
           "Control node x-coordinates must be real numbers.");
    assert(std::holds_alternative<type_real>(control_node_coords[0].y) &&
           "Control node y-coordinates must be real numbers.");
    assert(std::holds_alternative<type_real>(control_node_coords[0].z) &&
           "Control node z-coordinates must be real numbers.");

    for (int ix = 0; ix < ngll; ++ix) {
      for (int iy = 0; iy < ngll; ++iy) {
        for (int iz = 0; iz < ngll; ++iz) {
          const type_real xil = xi(ix);
          const type_real etal = xi(iy);
          const type_real zetal = xi(iz);
          const auto shape_function = specfem::shape_function::shape_function(
              xil, etal, zetal, ncontrol_nodes);

          type_real x = 0.0;
          type_real y = 0.0;
          type_real z = 0.0;
          for (int a = 0; a < ncontrol_nodes; ++a) {
            x += shape_function[a] *
                 std::get<type_real>(control_node_coords[a].x);
            y += shape_function[a] *
                 std::get<type_real>(control_node_coords[a].y);
            z += shape_function[a] *
                 std::get<type_real>(control_node_coords[a].z);
          }
          quadrature_coords(iz, iy, ix) = { x, y, z };
        }
      }
    }
  }
};

} // namespace specfem::mesh_entity_test

/**
 * @brief Test point counts for different mesh entity types in 3D element
 *
 * For a 5×5×5 GLL grid:
 * - Each face should contain 25 points (5×5)
 * - Each edge should contain 5 points
 * - Each corner should contain 1 point
 */
TEST(MeshEntity3D, ConnectionsPerNode) {

  specfem::mesh_entity::element element(5, 5, 5); // ngllz, nglly, ngllx

  for (const auto face : specfem::mesh_entity::dim3::faces) {
    const int npoints = element.number_of_points_on_orientation(face);
    EXPECT_EQ(npoints, 25) << "Face "
                           << specfem::mesh_entity::dim3::to_string(face)
                           << " should have 25 points (5x5) but has " << npoints
                           << " points." << std::endl;
  }

  for (const auto edge : specfem::mesh_entity::dim3::edges) {
    const int npoints = element.number_of_points_on_orientation(edge);
    EXPECT_EQ(npoints, 5) << "Edge "
                          << specfem::mesh_entity::dim3::to_string(edge)
                          << " should have 5 points but has " << npoints
                          << " points." << std::endl;
  }

  for (const auto corner : specfem::mesh_entity::dim3::corners) {
    const int npoints = element.number_of_points_on_orientation(corner);
    EXPECT_EQ(npoints, 1) << "Corner "
                          << specfem::mesh_entity::dim3::to_string(corner)
                          << " should have 1 point but has " << npoints
                          << " points." << std::endl;
  }
}

/**
 * @brief Configuration structure for parameterized mesh entity tests
 *
 * Defines test parameters for validating coordinate mappings of different
 * mesh entities (faces, edges, corners) in a spectral element. Each test
 * configuration specifies a mesh entity and its expected coordinate pattern.
 */
struct SingleElement3DTestConfig {
  /** @brief Mesh entity type (face, edge, or corner) to test */
  specfem::mesh_entity::dim3::type face;

  /** @brief Human-readable name for the test case */
  std::string name;

  /** @brief Expected coordinate pattern (may include wildcards) */
  specfem::mesh_entity_test::Coordinate3D expected;

  /**
   * @brief Constructor for test configuration
   *
   * @param face Mesh entity type to test
   * @param name Descriptive name for the test case
   * @param expected Expected coordinate pattern for all points on this entity
   */
  SingleElement3DTestConfig(
      const specfem::mesh_entity::dim3::type face, const std::string &name,
      const specfem::mesh_entity_test::Coordinate3D &expected)
      : face(face), name(name), expected(expected) {}
};

/**
 * @brief Stream output operator for test configuration
 *
 * Enables readable test names in GoogleTest output by printing the
 * configuration name when tests are executed.
 *
 * @param os Output stream
 * @param config Test configuration to output
 * @return Reference to the output stream
 */
std::ostream &operator<<(std::ostream &os,
                         const SingleElement3DTestConfig &config) {
  return os << config.name;
}

/**
 * @brief Parameterized test fixture for single element coordinate mapping
 *
 * This test fixture validates coordinate mappings for different mesh entities
 * (faces, edges, corners) within a single hexahedral spectral element. It uses
 * a unit cube element with 5×5×5 GLL quadrature points to test the accuracy
 * of coordinate transformations and mesh entity identification.
 *
 * The test setup includes:
 * - A unit cube element with corners at (0,0,0) and (1,1,1)
 * - 5×5×5 GLL quadrature grid
 * - Connection mapping for all mesh entity types
 *
 * @see SingleElement3DTestConfig
 */
class SingleElement3D
    : public ::testing::TestWithParam<SingleElement3DTestConfig> {
protected:
  /**
   * @brief Test setup (currently empty)
   *
   * All initialization is handled in the constructor. Override this
   * method if additional per-test setup is needed.
   */
  void SetUp() override {}

  /**
   * @brief Test cleanup (currently empty)
   *
   * Override this method if per-test cleanup is needed.
   */
  void TearDown() override {}

  /**
   * @brief Constructor with unit cube element setup
   *
   * Initializes the test fixture with a standard unit cube element
   * containing 5×5×5 GLL points. The element geometry is defined by
   * 8 control nodes forming a unit cube from (0,0,0) to (1,1,1).
   */
  SingleElement3D()
      : element(5,
                {
                    { 0.0, 0.0, 0.0 }, // 0: bottom-front-left
                    { 1.0, 0.0, 0.0 }, // 1: bottom-front-right
                    { 1.0, 1.0, 0.0 }, // 2: bottom-back-right
                    { 0.0, 1.0, 0.0 }, // 3: bottom-back-left
                    { 0.0, 0.0, 1.0 }, // 4: top-front-left
                    { 1.0, 0.0, 1.0 }, // 5: top-front-right
                    { 1.0, 1.0, 1.0 }, // 6: top-back-right
                    { 0.0, 1.0, 1.0 }  // 7: top-back-left
                }),
        mapping(5, 5, 5) {}

  /** @brief Hexahedral spectral element for testing */
  specfem::mesh_entity_test::Element8Node element;

  /** @brief 3D mesh entity connection mapping */
  specfem::mesh_entity::element<specfem::dimension::type::dim3> mapping;
};

/** @brief Alias for wildcard coordinate type */
using All = specfem::mesh_entity_test::Coordinate3D::All;

/**
 * @brief Parameterized test for mesh entity coordinate mapping validation
 *
 * This test validates that coordinate mappings work correctly for different
 * mesh entities within a spectral element. For each test configuration, it:
 *
 * 1. Retrieves the number of points associated with the mesh entity
 * 2. Maps each point index to grid coordinates (iz, iy, ix)
 * 3. Transforms grid coordinates to physical coordinates
 * 4. Verifies that physical coordinates match the expected pattern
 *
 * The test uses wildcard coordinates to handle entities where certain
 * components are constrained (e.g., x=0 for left face) while others vary.
 *
 * @note The test is parameterized over different mesh entities including
 *       faces, edges, and corners of the hexahedral element.
 */
TEST_P(SingleElement3D, MapCoordinatesTest) {
  const auto &config = GetParam();
  const int face_points = mapping.number_of_points_on_orientation(config.face);

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim3::corners,
                                     config.face)) {
    const auto [iz, iy, ix] = mapping.map_coordinates(config.face);
    const auto &corner_coords = element.quadrature_coords(iz, iy, ix);
    EXPECT_TRUE(corner_coords == config.expected)
        << "Corner " << config.name << " at" << corner_coords.to_string()
        << " does not match expected coordinate pattern "
        << config.expected.to_string() << std::endl;
    return; // Corner test complete
  }

  for (int ipoint = 0; ipoint < face_points; ++ipoint) {
    const auto [iz, iy, ix] = mapping.map_coordinates(config.face, ipoint);
    const auto &coord = element.quadrature_coords(iz, iy, ix);
    EXPECT_TRUE(coord == config.expected)
        << "Mesh entity " << config.name << " point " << ipoint << " at "
        << coord.to_string() << " does not match expected coordinate pattern "
        << config.expected.to_string() << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MeshEntity3D, SingleElement3D,
    ::testing::Values(
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::left, "LeftFace",
            specfem::mesh_entity_test::Coordinate3D(0.0, All(), All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::right, "RightFace",
            specfem::mesh_entity_test::Coordinate3D(1.0, All(), All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::front, "FrontFace",
            specfem::mesh_entity_test::Coordinate3D(All(), 0.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::back, "BackFace",
            specfem::mesh_entity_test::Coordinate3D(All(), 1.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom, "BottomFace",
            specfem::mesh_entity_test::Coordinate3D(All(), All(), 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top, "TopFace",
            specfem::mesh_entity_test::Coordinate3D(All(), All(), 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_right, "TopRightEdge",
            specfem::mesh_entity_test::Coordinate3D(1.0, All(), 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_left, "TopLeftEdge",
            specfem::mesh_entity_test::Coordinate3D(0.0, All(), 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_right, "BottomRightEdge",
            specfem::mesh_entity_test::Coordinate3D(1.0, All(), 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_left, "BottomLeftEdge",
            specfem::mesh_entity_test::Coordinate3D(0.0, All(), 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::front_right, "FrontRightEdge",
            specfem::mesh_entity_test::Coordinate3D(1.0, 0.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::front_left, "FrontLeftEdge",
            specfem::mesh_entity_test::Coordinate3D(0.0, 0.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::back_right, "BackRightEdge",
            specfem::mesh_entity_test::Coordinate3D(1.0, 1.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::back_left, "BackLeftEdge",
            specfem::mesh_entity_test::Coordinate3D(0.0, 1.0, All())),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::front_bottom, "FrontBottomEdge",
            specfem::mesh_entity_test::Coordinate3D(All(), 0.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::back_bottom, "BackBottomEdge",
            specfem::mesh_entity_test::Coordinate3D(All(), 1.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::front_top, "FrontTopEdge",
            specfem::mesh_entity_test::Coordinate3D(All(), 0.0, 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::back_top, "BackTopEdge",
            specfem::mesh_entity_test::Coordinate3D(All(), 1.0, 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_front_left,
            "BottomFrontLeftCorner",
            specfem::mesh_entity_test::Coordinate3D(0.0, 0.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_front_right,
            "BottomFrontRightCorner",
            specfem::mesh_entity_test::Coordinate3D(1.0, 0.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_front_left,
            "TopFrontLeftCorner",
            specfem::mesh_entity_test::Coordinate3D(0.0, 0.0, 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_front_right,
            "TopFrontRightCorner",
            specfem::mesh_entity_test::Coordinate3D(1.0, 0.0, 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_back_left,
            "BottomBackLeftCorner",
            specfem::mesh_entity_test::Coordinate3D(0.0, 1.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::bottom_back_right,
            "BottomBackRightCorner",
            specfem::mesh_entity_test::Coordinate3D(1.0, 1.0, 0.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_back_left,
            "TopBackLeftCorner",
            specfem::mesh_entity_test::Coordinate3D(0.0, 1.0, 1.0)),
        SingleElement3DTestConfig(
            specfem::mesh_entity::dim3::type::top_back_right,
            "TopBackRightCorner",
            specfem::mesh_entity_test::Coordinate3D(1.0, 1.0, 1.0))));
