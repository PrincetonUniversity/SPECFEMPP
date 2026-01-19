#include "../test_fixture.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/utilities.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace specfem::assembly_test {

struct TotalControlNodes {
  int nnodes_per_element; ///< Number of control nodes per element
  int nelements;          ///< Total number of elements in the mesh
  TotalControlNodes(int nnodes_per_element, int nelements)
      : nnodes_per_element(nnodes_per_element), nelements(nelements) {}
};

struct ControlNode3D {
  type_real x, y, z;
  int id;
  int element_id;
  int local_id;
  ControlNode3D(type_real x, type_real y, type_real z, int id, int element_id,
                int local_id)
      : x(x), y(y), z(z), id(id), element_id(element_id), local_id(local_id) {}
};

struct ExpectedControlNodes3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  TotalControlNodes total_control_nodes;
  std::vector<ControlNode3D> nodes;

  ExpectedControlNodes3D(TotalControlNodes total_control_nodes,
                         const std::initializer_list<ControlNode3D> nodes)
      : total_control_nodes(total_control_nodes), nodes(nodes) {}

  void check(const specfem::assembly::mesh_impl::control_nodes<dimension>
                 &control_nodes) const {
    ASSERT_EQ(control_nodes.nspec, total_control_nodes.nelements)
        << "Total number of elements mismatch. "
        << "Expected: " << total_control_nodes.nelements << ", "
        << "Got: " << control_nodes.nspec << std::endl;
    ASSERT_EQ(control_nodes.ngnod, total_control_nodes.nnodes_per_element)
        << "Total number of control nodes per element mismatch. "
        << "Expected: " << total_control_nodes.nnodes_per_element << ", "
        << "Got: " << control_nodes.ngnod << std::endl;

    // Make sure all views are allocated
    ASSERT_TRUE(control_nodes.control_node_index.extent(0) ==
                control_nodes.nspec)
        << "Control node index extent 0 mismatch.";
    ASSERT_TRUE(control_nodes.control_node_index.extent(1) ==
                control_nodes.ngnod)
        << "Control node index extent 1 mismatch.";
    ASSERT_TRUE(control_nodes.control_node_coordinates.extent(0) ==
                control_nodes.nspec)
        << "Control node coordinates extent 0 mismatch.";
    ASSERT_TRUE(control_nodes.control_node_coordinates.extent(1) ==
                control_nodes.ngnod)
        << "Control node coordinates extent 1 mismatch.";
    ASSERT_TRUE(control_nodes.control_node_coordinates.extent(2) == 3)
        << "Control node coordinates extent 2 mismatch.";

    ASSERT_TRUE(control_nodes.h_control_node_index.extent(0) ==
                control_nodes.nspec)
        << "Control node index extent 0 mismatch.";
    ASSERT_TRUE(control_nodes.h_control_node_index.extent(1) ==
                control_nodes.ngnod)
        << "Control node index extent 1 mismatch.";
    ASSERT_TRUE(control_nodes.h_control_node_coordinates.extent(0) ==
                control_nodes.nspec)
        << "Control node coordinates extent 0 mismatch.";
    ASSERT_TRUE(control_nodes.h_control_node_coordinates.extent(1) ==
                control_nodes.ngnod)
        << "Control node coordinates extent 1 mismatch.";
    ASSERT_TRUE(control_nodes.h_control_node_coordinates.extent(2) == 3)
        << "Control node coordinates extent 2 mismatch.";

    for (const auto &expected_node : nodes) {
      const int id = expected_node.id;

      if (id < 0 || id >= control_nodes.ngnod * control_nodes.nspec) {
        FAIL() << "Expected control node ID " << id << " is out of range."
               << std::endl;

        return;
      }

      EXPECT_EQ(control_nodes.h_control_node_index(expected_node.element_id,
                                                   expected_node.local_id),
                id)
          << "Control node ID mismatch for element " << expected_node.element_id
          << " local node " << expected_node.local_id << ". "
          << "Expected: " << expected_node.id << ", "
          << "Got: "
          << control_nodes.h_control_node_index(expected_node.element_id,
                                                expected_node.local_id)
          << std::endl;

      if (!specfem::utilities::is_close(
              control_nodes.h_control_node_coordinates(
                  expected_node.element_id, expected_node.local_id, 0),
              expected_node.x) ||
          !specfem::utilities::is_close(
              control_nodes.h_control_node_coordinates(
                  expected_node.element_id, expected_node.local_id, 1),
              expected_node.y) ||
          !specfem::utilities::is_close(
              control_nodes.h_control_node_coordinates(
                  expected_node.element_id, expected_node.local_id, 2),
              expected_node.z)) {
        FAIL() << "Control node ID " << id << " coordinates do not match. "
               << "Expected: (" << expected_node.x << ", " << expected_node.y
               << ", " << expected_node.z << "), "
               << "Got: ("
               << control_nodes.h_control_node_coordinates(
                      expected_node.element_id, expected_node.local_id, 0)
               << ", "
               << control_nodes.h_control_node_coordinates(
                      expected_node.element_id, expected_node.local_id, 1)
               << ", "
               << control_nodes.h_control_node_coordinates(
                      expected_node.element_id, expected_node.local_id, 2)
               << ")" << std::endl;
      }

      SUCCEED() << "All expected control nodes are present and correct."
                << std::endl;
    }
  }
};

} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

// Expected control nodes for specific test cases
static const std::unordered_map<std::string, ExpectedControlNodes3D>
    expected_control_nodes_map = {
      { "EightNodeElastic",
        ExpectedControlNodes3D(
            TotalControlNodes(8, 8),
            { // Domain corner
              ControlNode3D(0.0, 0.0, 0.0, 4, 4, 4),
              // Shared nodes
              ControlNode3D(50000.0, 40000.0, -30000.0, 62, 0, 6),
              ControlNode3D(50000.0, 40000.0, -30000.0, 62, 2, 5) }) }
    };

TEST_P(Assembly3DTest, ControlNodes) {
  const auto &param_name = GetParam();
  if (expected_control_nodes_map.find(param_name) ==
      expected_control_nodes_map.end()) {
    GTEST_SKIP() << "No expected control nodes defined for parameter: "
                 << param_name;
    return;
  }

  const auto &assembly = getAssembly();
  const auto &control_nodes = assembly.mesh;
  const auto &expected_nodes = expected_control_nodes_map.at(param_name);

  expected_nodes.check(control_nodes);
}
