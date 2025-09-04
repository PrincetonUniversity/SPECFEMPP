
#include "../../MPI_environment.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

const static std::unordered_map<std::string, std::string> mesh_files = {
  { "Circular mesh", "data/mesh/circular_mesh/database.bin" }
};

const static std::unordered_map<std::string, std::map<int, std::vector<int> > >
    expected_adjacency_graph = {
      { "Circular mesh", { { 37, { 39, 38, 64, 63, 36, 83, 2, 3, 1 } } } }
    };

class CheckConnections : public ::testing::TestWithParam<std::string> {
protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_P(CheckConnections, Test) {
  const std::string mesh_name = GetParam();

  const std::string mesh_file = mesh_files.at(mesh_name);

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto mesh =
      specfem::io::read_2d_mesh(mesh_file, specfem::enums::elastic_wave::psv,
                                specfem::enums::electromagnetic_wave::te, mpi);

  const auto &adjacency_graph = mesh.adjacency_graph;

  ASSERT_FALSE(adjacency_graph.empty()) << "Adjacency graph is empty";

  const auto &g = adjacency_graph.graph();

  EXPECT_NO_THROW(adjacency_graph.assert_symmetry());

  const auto expected_adjacency = expected_adjacency_graph.at(mesh_name);

  for (const auto &[ispec, expected_neighbors] : expected_adjacency) {
    const auto out_edges = boost::out_edges(ispec, g);
    std::vector<int> computed_neighbors;
    for (auto edge_it = out_edges.first; edge_it != out_edges.second;
         ++edge_it) {
      const int neighbor = boost::target(*edge_it, g);
      computed_neighbors.push_back(neighbor);
    }

    {
      std::ostringstream message;
      message << "Length mismatch for node " << ispec << ":\n"
              << "Expected length: " << expected_neighbors.size() << "\n"
              << "Actual length: " << computed_neighbors.size() << "\n";

      EXPECT_EQ(computed_neighbors.size(), expected_neighbors.size())
          << message.str();
    }

    {
      std::sort(computed_neighbors.begin(), computed_neighbors.end());
      auto expected_sorted = expected_neighbors;
      std::sort(expected_sorted.begin(), expected_sorted.end());

      std::ostringstream message;
      message << "Adjacency list mismatch for node " << ispec << ":\n"
              << "Expected: ";
      for (const auto &n : expected_sorted) {
        message << n << " ";
      }
      message << "\nActual: ";
      for (const auto &n : computed_neighbors) {
        message << n << " ";
      }
      message << "\n";

      EXPECT_EQ(computed_neighbors, expected_sorted) << message.str();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AdjacencyGraphIrregularMesh, CheckConnections,
                         ::testing::Values("Circular mesh"
                                           // Add more mesh names here as needed
                                           ));
