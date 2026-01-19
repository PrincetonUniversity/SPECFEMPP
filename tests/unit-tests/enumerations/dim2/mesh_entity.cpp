/**
 * @file mesh_entity.cpp
 * @brief Unit tests for 2D spectral element connectivity mapping
 *
 * This file contains comprehensive unit tests for the 2D element connectivity
 * functionality in the SPECFEM++ spectral element framework. The tests verify
 * that coordinate mappings between mesh entities (edges, corners) and
 * their corresponding grid points work correctly for a single quadrilateral
 * element.
 *
 * The tests cover:
 * - Edge coordinate mapping for all 4 quadrilateral edges
 * - Corner coordinate mapping for all 4 quadrilateral corners
 * - Point count validation for different mesh entity types
 * - Coordinate transformation accuracy using spectral element shape functions
 *
 * @see specfem::mesh_entity::dim2
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
 * @brief Flexible coordinate representation for 2D spectral element testing
 *
 * This structure provides a coordinate system that supports both exact values
 * and wildcards (All) for flexible coordinate matching in tests. The wildcard
 * functionality is essential for testing mesh entity mappings where certain
 * coordinate components vary while others remain constant.
 *
 * @code
 * // Exact coordinate specification
 * Coordinate corner(1.0, 0.0);
 *
 * // Wildcard coordinate for edge testing (x=1, z varies)
 * Coordinate right_edge(1.0, Coordinate::All());
 *
 * // Test coordinate equality with wildcards
 * Coordinate test_point(1.0, 0.5);
 * assert(test_point == right_edge); // True: x matches, z is wildcard
 * @endcode
 */
struct Coordinate2D {

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

  /** @brief Z-coordinate: either a real value or wildcard */
  std::variant<type_real, All> z = static_cast<type_real>(0.0);

  /**
   * @brief Default constructor
   *
   * Creates a coordinate at the origin (0, 0).
   */
  Coordinate2D() = default;

  /**
   * @brief Generic constructor with type-safe coordinate assignment
   *
   * Constructs a coordinate with flexible type handling, supporting both
   * floating-point values and wildcard (All) types for any coordinate
   * component. This template-based approach ensures type safety while allowing
   * maximum flexibility in coordinate specification.
   *
   * @tparam X Type of x-coordinate (type_real or All)
   * @tparam Z Type of z-coordinate (type_real or All)
   * @param x_val X-coordinate value or wildcard
   * @param z_val Z-coordinate value or wildcard
   *
   * @code
   * // Various coordinate construction examples
   * Coordinate origin(0.0, 0.0);
   * Coordinate corner(1.0, 1.0);
   * Coordinate x_edge(1.0, Coordinate::All());
   * @endcode
   */
  template <typename X = type_real, typename Z = type_real,
            typename std::enable_if_t<
                std::is_floating_point_v<X> || std::is_same_v<X, All>, int> = 0,
            typename std::enable_if_t<
                std::is_floating_point_v<Z> || std::is_same_v<Z, All>, int> = 0>
  Coordinate2D(X x_val, Z z_val)
      : x([&]() -> std::variant<type_real, All> {
          if constexpr (std::is_same_v<X, All>) {
            return All{};
          } else {
            return static_cast<type_real>(x_val);
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
   * Constructs a coordinate from an initializer list of exactly 2 values
   * representing x and z coordinates respectively.
   *
   * @param list Initializer list containing exactly 2 coordinate values
   * @throws std::runtime_error if the list doesn't contain exactly 2 values
   *
   * @code
   * Coordinate corner = {1.0, 1.0};
   * Coordinate origin = {0.0, 0.0};
   * @endcode
   */
  Coordinate2D(const std::initializer_list<type_real> &list) {
    if (list.size() != 2) {
      throw std::runtime_error(
          "Coordinate assignment requires exactly 2 values.");
    }
    auto it = list.begin();
    x = *it++;
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
   * Coordinate exact(1.0, 0.5);
   * Coordinate pattern(1.0, Coordinate::All());
   * assert(exact == pattern); // True: x matches, z is wildcard
   * @endcode
   */
  bool operator==(const Coordinate2D &other) const {
    bool x_equal = false;
    if (std::holds_alternative<All>(x) ||
        std::holds_alternative<All>(other.x)) {
      x_equal = true; // Wildcard matches any value
    } else {
      x_equal = (specfem::utilities::is_close(std::get<type_real>(x),
                                              std::get<type_real>(other.x)));
    }

    bool z_equal = false;
    if (std::holds_alternative<All>(z) ||
        std::holds_alternative<All>(other.z)) {
      z_equal = true; // Wildcard matches any value
    } else {
      z_equal = (specfem::utilities::is_close(std::get<type_real>(z),
                                              std::get<type_real>(other.z)));
    }

    return x_equal && z_equal;
  }

  /**
   * @brief Convert coordinate to string representation
   *
   * @return String representation of the coordinate
   */
  std::string to_string() const {
    std::ostringstream os;
    os << "(";
    if (std::holds_alternative<All>(x)) {
      os << "*";
    } else {
      os << std::get<type_real>(x);
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
 * @brief 4-node quadrilateral spectral element for connection mapping tests
 *
 * This class represents a 2D quadrilateral spectral element with 4 control
 * nodes and Gauss-Lobatto-Legendre (GLL) quadrature points. It provides the
 * geometric foundation for testing coordinate mapping between mesh entities and
 * their corresponding grid points in the spectral element method.
 *
 * The element uses isoparametric mapping with bilinear shape functions to
 * transform between the reference element coordinates (\f$\xi, \zeta \in
 * [-1,1]^2\f$) and physical coordinates. The 4 control nodes define the element
 * geometry:
 *
 * @code
 * Node numbering (standard quadrilateral convention):
 *  3----------2
 *  |          |
 *  |          |
 *  |          |
 *  0----------1
 * @endcode
 *
 * @see specfem::shape_function::shape_function
 * @see specfem::quadrature::gll::gll
 */
struct Element4Node {

  /** @brief Dimension tag for 2D spectral elements */
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

  /** @brief Number of control nodes in quadrilateral element */
  constexpr static int ncontrol_nodes = 4;

  /**
   * @brief Gauss-Lobatto-Legendre quadrature object
   *
   * Provides quadrature points and weights for numerical integration
   * over the reference element domain [-1, 1]².
   */
  specfem::quadrature::gll::gll quadrature;

  /**
   * @brief Number of GLL points in each spatial dimension
   *
   * For square elements, this is the same in both dimensions (ngllx = ngllz).
   */
  int ngll;

  /**
   * @brief Physical coordinates of the 4 quadrilateral control nodes
   *
   * These coordinates define the element geometry in physical space.
   * The nodes are ordered according to standard quadrilateral convention
   * with node 0 at the origin and node 2 at the opposite corner.
   */
  std::array<Coordinate2D, ncontrol_nodes> control_node_coords;

  /**
   * @brief Physical coordinates of all GLL quadrature points
   *
   * 2D Kokkos view storing the transformed coordinates of every GLL point
   * within the element. Layout: quadrature_coords(iz, ix) where
   * indices correspond to the ζ and ξ directions respectively.
   */
  Kokkos::View<Coordinate2D **, Kokkos::LayoutLeft, Kokkos::HostSpace>
      quadrature_coords;

  /**
   * @brief Default constructor
   *
   * Creates an uninitialized element. Use the parameterized constructor
   * for proper element setup with quadrature points and control nodes.
   */
  Element4Node() = default;

  /**
   * @brief Parameterized constructor with control node specification
   *
   * Creates a quadrilateral spectral element with the specified number of GLL
   * points and control node coordinates. Automatically computes all quadrature
   * point coordinates using isoparametric transformation.
   *
   * @param ngll Number of GLL quadrature points per dimension
   * @param coords Initializer list of 4 control node coordinates in standard
   * order
   *
   * @throws std::runtime_error if coords doesn't contain exactly 4 coordinates
   *
   * @code
   * // Create unit square element with 5 GLL points per dimension
   * Element4Node unit_square(5, {
   *     {0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}
   * });
   * @endcode
   */
  Element4Node(const int ngll,
               const std::initializer_list<Coordinate2D> &coords)
      : ngll(ngll), quadrature(0.0, 0.0, ngll),
        quadrature_coords("quad_coords", ngll, ngll), control_node_coords() {
    if (coords.size() != ncontrol_nodes) {
      throw std::runtime_error(
          "Element4Node requires exactly 4 control node coordinates.");
    }
    std::copy(coords.begin(), coords.end(), control_node_coords.begin());
    compute_coordinates();
  }

  /**
   * @brief Compute physical coordinates for all GLL quadrature points
   *
   * Uses isoparametric transformation with bilinear quadrilateral shape
   * functions to map from reference element coordinates (\f$\xi, \zeta\f$) to
   * physical coordinates. The transformation is:
   *
   * \f[
   * \mathbf{x}(\xi, \zeta) = \sum_{a=1}^{4} N_a(\xi, \zeta)
   * \mathbf{x}_a
   * \f]
   *
   * where \f$N_a\f$ are the bilinear shape functions and \f$\mathbf{x}_a\f$
   * are the control node coordinates.
   *
   * @note This function is automatically called by the constructor and should
   *       be called manually only if control node coordinates are modified.
   */
  void compute_coordinates() {
    const auto &xi = quadrature.get_hxi();

    assert(std::holds_alternative<type_real>(control_node_coords[0].x) &&
           "Control node x-coordinates must be real numbers.");
    assert(std::holds_alternative<type_real>(control_node_coords[0].z) &&
           "Control node z-coordinates must be real numbers.");

    for (int ix = 0; ix < ngll; ++ix) {
      for (int iz = 0; iz < ngll; ++iz) {
        const type_real xil = xi(ix);
        const type_real zetal = xi(iz);
        const auto shape_function =
            specfem::shape_function::shape_function(xil, zetal, ncontrol_nodes);

        type_real x = 0.0;
        type_real z = 0.0;
        for (int a = 0; a < ncontrol_nodes; ++a) {
          x +=
              shape_function[a] * std::get<type_real>(control_node_coords[a].x);
          z +=
              shape_function[a] * std::get<type_real>(control_node_coords[a].z);
        }
        quadrature_coords(iz, ix) = { x, z };
      }
    }
  }
};

} // namespace specfem::mesh_entity_test

/**
 * @brief Test point counts for different mesh entity types in 2D element
 *
 * For a 5×5 GLL grid:
 * - Each edge should contain 5 points
 * - Each corner should contain 1 point
 */
TEST(MeshEntity2D, ConnectionsPerNode) {

  specfem::mesh_entity::element<specfem::dimension::type::dim2> element(
      5, 5); // ngllz, ngllx

  for (const auto edge : specfem::mesh_entity::dim2::edges) {
    const int npoints = element.number_of_points_on_orientation(edge);
    EXPECT_EQ(npoints, 5) << "Edge "
                          << specfem::mesh_entity::dim2::to_string(edge)
                          << " should have 5 points but has " << npoints
                          << " points." << std::endl;
  }

  for (const auto corner : specfem::mesh_entity::dim2::corners) {
    const int npoints = element.number_of_points_on_orientation(corner);
    EXPECT_EQ(npoints, 1) << "Corner "
                          << specfem::mesh_entity::dim2::to_string(corner)
                          << " should have 1 point but has " << npoints
                          << " points." << std::endl;
  }
}

/**
 * @brief Configuration structure for parameterized mesh entity tests
 *
 * Defines test parameters for validating coordinate mappings of different
 * mesh entities (edges, corners) in a 2D spectral element. Each test
 * configuration specifies a mesh entity and its expected coordinate pattern.
 */
struct SingleElement2DTestConfig {
  /** @brief Mesh entity type (edge or corner) to test */
  specfem::mesh_entity::dim2::type entity;

  /** @brief Human-readable name for the test case */
  std::string name;

  /** @brief Expected coordinate pattern (may include wildcards) */
  specfem::mesh_entity_test::Coordinate2D expected;

  /**
   * @brief Constructor for test configuration
   *
   * @param entity Mesh entity type to test
   * @param name Descriptive name for the test case
   * @param expected Expected coordinate pattern for all points on this entity
   */
  SingleElement2DTestConfig(
      const specfem::mesh_entity::dim2::type entity, const std::string &name,
      const specfem::mesh_entity_test::Coordinate2D &expected)
      : entity(entity), name(name), expected(expected) {}
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
                         const SingleElement2DTestConfig &config) {
  return os << config.name;
}

/**
 * @brief Parameterized test fixture for single 2D element coordinate mapping
 *
 * This test fixture validates coordinate mappings for different mesh entities
 * (edges, corners) within a single quadrilateral spectral element. It uses
 * a unit square element with 5×5 GLL quadrature points to test the accuracy
 * of coordinate transformations and mesh entity identification.
 *
 * The test setup includes:
 * - A unit square element with corners at (0,0) and (1,1)
 * - 5×5 GLL quadrature grid
 * - Connection mapping for all mesh entity types
 *
 * @see SingleElement2DTestConfig
 */
class SingleElement2D
    : public ::testing::TestWithParam<SingleElement2DTestConfig> {
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
   * @brief Constructor with unit square element setup
   *
   * Initializes the test fixture with a standard unit square element
   * containing 5×5 GLL points. The element geometry is defined by
   * 4 control nodes forming a unit square from (0,0) to (1,1).
   */
  SingleElement2D()
      : element(5,
                {
                    { 0.0, 0.0 }, // 0: bottom-left
                    { 1.0, 0.0 }, // 1: bottom-right
                    { 1.0, 1.0 }, // 2: top-right
                    { 0.0, 1.0 }  // 3: top-left
                }),
        mapping(5, 5) {}

  /** @brief Quadrilateral spectral element for testing */
  specfem::mesh_entity_test::Element4Node element;

  /** @brief 2D mesh entity connection mapping */
  specfem::mesh_entity::element<specfem::dimension::type::dim2> mapping;
};

/** @brief Alias for wildcard coordinate type */
using All = specfem::mesh_entity_test::Coordinate2D::All;

/**
 * @brief Parameterized test for mesh entity coordinate mapping validation
 *
 * This test validates that coordinate mappings work correctly for different
 * mesh entities within a 2D spectral element. For each test configuration, it:
 *
 * 1. Retrieves the number of points associated with the mesh entity
 * 2. Maps each point index to grid coordinates (iz, ix)
 * 3. Transforms grid coordinates to physical coordinates
 * 4. Verifies that physical coordinates match the expected pattern
 *
 * The test uses wildcard coordinates to handle entities where certain
 * components are constrained (e.g., x=0 for left edge) while others vary.
 *
 * @note The test is parameterized over different mesh entities including
 *       edges and corners of the quadrilateral element.
 */
TEST_P(SingleElement2D, MapCoordinatesTest) {
  const auto &config = GetParam();
  const int entity_points =
      mapping.number_of_points_on_orientation(config.entity);

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                     config.entity)) {
    const auto [iz, ix] = mapping.map_coordinates(config.entity);
    const auto &corner_coords = element.quadrature_coords(iz, ix);
    EXPECT_TRUE(corner_coords == config.expected)
        << "Corner " << config.name << " at " << corner_coords.to_string()
        << " does not match expected coordinate pattern "
        << config.expected.to_string() << std::endl;
    return; // Corner test complete
  }

  for (int ipoint = 0; ipoint < entity_points; ++ipoint) {
    const auto [iz, ix] = mapping.map_coordinates(config.entity, ipoint);
    const auto &coord = element.quadrature_coords(iz, ix);
    EXPECT_TRUE(coord == config.expected)
        << "Mesh entity " << config.name << " point " << ipoint << " at "
        << coord.to_string() << " does not match expected coordinate pattern "
        << config.expected.to_string() << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MeshEntity2D, SingleElement2D,
    ::testing::Values(
        // Edge tests
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::left, "LeftEdge",
            specfem::mesh_entity_test::Coordinate2D(0.0, All())),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::right, "RightEdge",
            specfem::mesh_entity_test::Coordinate2D(1.0, All())),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::bottom, "BottomEdge",
            specfem::mesh_entity_test::Coordinate2D(All(), 0.0)),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::top, "TopEdge",
            specfem::mesh_entity_test::Coordinate2D(All(), 1.0)),
        // Corner tests
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::bottom_left, "BottomLeftCorner",
            specfem::mesh_entity_test::Coordinate2D(0.0, 0.0)),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::bottom_right, "BottomRightCorner",
            specfem::mesh_entity_test::Coordinate2D(1.0, 0.0)),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::top_right, "TopRightCorner",
            specfem::mesh_entity_test::Coordinate2D(1.0, 1.0)),
        SingleElement2DTestConfig(
            specfem::mesh_entity::dim2::type::top_left, "TopLeftCorner",
            specfem::mesh_entity_test::Coordinate2D(0.0, 1.0))));
