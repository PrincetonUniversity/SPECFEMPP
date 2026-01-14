
#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"
#include "specfem_setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

struct Connection {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  int ispec;
  int jspec;
  specfem::connections::type connection;
  specfem::mesh_entity::dim3::type orientation;

  Connection(int ispec, int jspec, specfem::connections::type connection,
             specfem::mesh_entity::dim3::type orientation)
      : ispec(ispec), jspec(jspec), connection(connection),
        orientation(orientation) {}

  ~Connection() = default;

  void expect_in(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const {
    const auto &g = adjacency_graph.graph();

    const auto [edge_, exists] = boost::edge(ispec, jspec, g);
    if (!exists) {
      std::ostringstream msg;
      msg << "Failed expected adjacency between elements " << ispec << " and "
          << jspec << ":\n";
      msg << "  Adjacency graph did not contain edge between " << ispec
          << " and " << jspec << "\n";
      FAIL() << msg.str();
    }
    const auto edge = g[edge_];
    if (edge.connection != connection) {
      std::ostringstream msg;
      msg << "Failed expected adjacency between elements " << ispec << " and "
          << jspec << ":\n";
      msg << "  Found connection type "
          << specfem::connections::to_string(edge.connection)
          << " for edge between " << ispec << " and " << jspec << "\n";
      FAIL() << msg.str();
    }
    if (edge.orientation != orientation) {
      std::ostringstream msg;
      msg << "Failed expected adjacency between elements " << ispec << " and "
          << jspec << ":\n";
      msg << "  Found orientation "
          << specfem::mesh_entity::dim3::to_string(edge.orientation)
          << " for edge between " << ispec << " and " << jspec << "\n";
      FAIL() << msg.str();
    }
  }
};

struct ExpectedAdjacency3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  int nelements;                       ///< Total number of elements in the mesh
  std::vector<Connection> connections; ///< List of expected connections

  ExpectedAdjacency3D(int nelements,
                      const std::initializer_list<Connection> &connections)
      : nelements(nelements), connections(connections) {}

  void check(
      const specfem::mesh::adjacency_graph<dimension> &adjacency_graph) const {
    // Verify total number of elements
    if (adjacency_graph.nspec != nelements) {
      FAIL() << "Total number of elements mismatch. "
             << "Expected: " << nelements << ", "
             << "Got: " << adjacency_graph.nspec << std::endl;
    }

    // Check each expected connection
    for (const auto &expected_connection : connections) {
      expected_connection.expect_in(adjacency_graph);
    }
    SUCCEED() << "All expected connections are present and correct."
              << std::endl;
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

static const std::unordered_map<std::string, ExpectedAdjacency3D>
    expected_adjacency_map = { {
        "EightNodeElastic",
        ExpectedAdjacency3D(
            8,
            { Connection(0, 1, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::right),
              Connection(1, 0, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::left),
              Connection(0, 2, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::back),
              Connection(2, 0, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::front),
              Connection(0, 4, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::top),
              Connection(4, 0, specfem::connections::type::strongly_conforming,
                         specfem::mesh_entity::dim3::type::bottom) })
        // Add more test cases as needed
    } };

TEST_P(Mesh3DTest, AdjacencyGraph) {
  const auto &param_name = GetParam();
  if (expected_adjacency_map.find(param_name) == expected_adjacency_map.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name
                 << std::endl;
    return;
  }

  const auto &mesh = getMesh();
  const auto &adjacency_graph = mesh.adjacency_graph;
  const auto &expected = expected_adjacency_map.at(param_name);
  expected.check(adjacency_graph);
}
