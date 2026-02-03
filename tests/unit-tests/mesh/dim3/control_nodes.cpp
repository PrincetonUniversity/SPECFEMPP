
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

/**
 * @brief Represents the total number of nodes and elements in the mesh.
 *
 */
struct TotalNodes {
  int nnodes;    ///< Total number of nodes in the mesh
  int nelements; ///< Total number of elements in the mesh

  TotalNodes(int nnodes, int nx, int ny, int nz)
      : nnodes(nnodes), nelements(nx * ny * nz) {}
};

/**
 * @brief Represents the size of the domain in 3D space.
 *
 */
struct DomainSize {
  type_real x_min, x_max; ///< Domain size in x-direction
  type_real y_min, y_max; ///< Domain size in y-direction
  type_real z_min, z_max; ///< Domain size in z-direction

  DomainSize(type_real x_min, type_real x_max, type_real y_min, type_real y_max,
             type_real z_min, type_real z_max)
      : x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max), z_min(z_min),
        z_max(z_max) {}
};

/**
 * @brief Represents a control node in 3D space.
 *
 */
struct ControlNode3D {
  type_real x, y, z; ///< Coordinates of the control node
  int id;            ///< Unique identifier for the control node
  int element_id;    ///< Identifier for the element to which the control node
                     ///< belongs
  int local_id; ///< Local identifier for the control node within its element
  ControlNode3D(type_real x, type_real y, type_real z, int id, int element_id,
                int local_id)
      : x(x), y(y), z(z), id(id), element_id(element_id), local_id(local_id) {}
};

// Expected control nodes for a specific test case
struct ExpectedControlNodes3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3; ///< Dimension of the control nodes
  TotalNodes total_nodes;             ///< Total nodes and elements in the mesh
  DomainSize domain_size;             ///< Domain size in 3D space
  std::vector<ControlNode3D> nodes;   ///< List of expected control nodes (These
                                    ///< are hard-coded for specific test cases)

  ExpectedControlNodes3D(TotalNodes total_nodes, DomainSize domain_size,
                         std::initializer_list<ControlNode3D> nodes)
      : total_nodes(total_nodes), domain_size(domain_size), nodes(nodes) {}

  /**
   * @brief Check if all the expected control nodes are present in the given
   * control nodes.
   *
   * @param control_nodes The control nodes to check against.
   * @return true if all expected control nodes are present, false otherwise.
   */
  void
  check(const specfem::mesh::control_nodes<dimension> &control_nodes) const {
    // Verify that the control nodes object has the expected number of nodes
    if (control_nodes.nnodes != total_nodes.nnodes) {
      FAIL() << "Total number of control nodes mismatch. "
             << "Expected: " << total_nodes.nnodes << ", "
             << "Got: " << control_nodes.nnodes << std::endl;
    }

    // Verify that the control nodes object has the expected number of
    // elements
    if (control_nodes.nspec != total_nodes.nelements) {
      FAIL() << "Total number of elements mismatch. "
             << "Expected: " << total_nodes.nelements << ", "
             << "Got: " << control_nodes.nspec << std::endl;
    }

    // Verify that the domain size matches the expected domain size
    if (!specfem::utilities::is_close(control_nodes.xmin, domain_size.x_min) ||
        !specfem::utilities::is_close(control_nodes.xmax, domain_size.x_max) ||
        !specfem::utilities::is_close(control_nodes.ymin, domain_size.y_min) ||
        !specfem::utilities::is_close(control_nodes.ymax, domain_size.y_max) ||
        !specfem::utilities::is_close(control_nodes.zmin, domain_size.z_min) ||
        !specfem::utilities::is_close(control_nodes.zmax, domain_size.z_max)) {
      FAIL() << "Domain size mismatch. "
             << "Expected: ["
             << "(" << domain_size.x_min << ", " << domain_size.x_max << "), "
             << "(" << domain_size.y_min << ", " << domain_size.y_max << "), "
             << "(" << domain_size.z_min << ", " << domain_size.z_max << ")]"
             << ", "
             << "Got: ["
             << "(" << control_nodes.xmin << ", " << control_nodes.xmax << "), "
             << "(" << control_nodes.ymin << ", " << control_nodes.ymax << "), "
             << "(" << control_nodes.zmin << ", " << control_nodes.zmax << ")]"
             << std::endl;
    }

    for (const auto &expected_node : nodes) {
      const int id = expected_node.id;

      // Check if the control node ID is within valid range
      if (id < 0 || id >= control_nodes.nnodes) {
        FAIL() << "Control node ID " << id << " is out of range." << std::endl;
      }

      // Check if the element ID is within valid range
      if (expected_node.element_id < 0 ||
          expected_node.element_id >= control_nodes.nspec) {
        FAIL() << "Element ID " << expected_node.element_id
               << " is out of range." << std::endl;
      }

      // Check if the local ID is within valid range
      if (expected_node.local_id < 0 ||
          expected_node.local_id >= control_nodes.ngnod) {
        FAIL() << "Local node ID " << expected_node.local_id
               << " is out of range." << std::endl;
      }

      // Verify that the control node ID matches the expected ID
      if (expected_node.id !=
          control_nodes.control_node_index(expected_node.element_id,
                                           expected_node.local_id)) {
        FAIL() << "Control node ID mismatch for element "
               << expected_node.element_id << " local node "
               << expected_node.local_id << ". "
               << "Expected: " << expected_node.id << ", "
               << "Got: "
               << control_nodes.control_node_index(expected_node.element_id,
                                                   expected_node.local_id)
               << std::endl;
      }

      // Verify that the coordinates match the expected coordinates
      if (!specfem::utilities::is_close(control_nodes.coordinates(id, 0),
                                        expected_node.x) ||
          !specfem::utilities::is_close(control_nodes.coordinates(id, 1),
                                        expected_node.y) ||
          !specfem::utilities::is_close(control_nodes.coordinates(id, 2),
                                        expected_node.z)) {
        FAIL() << "Control node ID " << id << " coordinates do not match. "
               << "Expected: (" << expected_node.x << ", " << expected_node.y
               << ", " << expected_node.z << "), "
               << "Got: (" << control_nodes.coordinates(id, 0) << ", "
               << control_nodes.coordinates(id, 1) << ", "
               << control_nodes.coordinates(id, 2) << ")" << std::endl;
      }
    }
    SUCCEED() << "All expected control nodes are present and correct."
              << std::endl;

    return;
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

// Expected control nodes for specific test cases
static const std::unordered_map<std::string, ExpectedControlNodes3D>
    expected_control_nodes_map = {
      { "EightNodeElastic",
        ExpectedControlNodes3D(
            TotalNodes(125, 2, 2, 2),
            DomainSize(0.0, 100000.0, 0.0, 80000.0, -60000.0, 0.0),
            { // Domain corner
              ControlNode3D(0.0, 0.0, 0.0, 4, 4, 4),
              // Shared nodes
              ControlNode3D(50000.0, 40000.0, -30000.0, 62, 0, 6),
              ControlNode3D(50000.0, 40000.0, -30000.0, 62, 2, 5) }) }
    };

TEST_P(Mesh3DTest, ControlNodes) {
  const auto &param_name = GetParam();
  if (expected_control_nodes_map.find(param_name) ==
      expected_control_nodes_map.end()) {
    GTEST_SKIP() << "No expected control nodes defined for parameter: "
                 << param_name;
    return;
  }

  const auto &mesh = getMesh();
  const auto &control_nodes = mesh.control_nodes;
  const auto &expected_nodes = expected_control_nodes_map.at(param_name);

  expected_nodes.check(control_nodes);
}
