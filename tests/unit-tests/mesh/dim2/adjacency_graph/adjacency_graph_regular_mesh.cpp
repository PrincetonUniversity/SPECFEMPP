
#include "SPECFEM_Environment.hpp"
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

constexpr static int nx =
    80; // See Par_file in data/dim2/regular_mesh/provenance
constexpr static int nz =
    60; // See Par_file in data/dim2/regular_mesh/provenance

const static std::unordered_map<int, std::vector<int> > expected_adjacency{
  // Interior node example
  { (nz / 2) * nx + (nx / 2),
    { (nz / 2 - 1) * nx + (nx / 2 - 1), (nz / 2 - 1) * nx + (nx / 2),
      (nz / 2 - 1) * nx + (nx / 2 + 1), (nz / 2) * nx + (nx / 2 - 1),
      (nz / 2) * nx + (nx / 2 + 1), (nz / 2 + 1) * nx + (nx / 2 - 1),
      (nz / 2 + 1) * nx + (nx / 2), (nz / 2 + 1) * nx + (nx / 2 + 1) } },
  // Edge node example
  { (nz / 2) * nx + 0,
    { (nz / 2 - 1) * nx + 0, (nz / 2 - 1) * nx + 1, (nz / 2) * nx + 1,
      (nz / 2 + 1) * nx + 0, (nz / 2 + 1) * nx + 1 } },
  // Corner node example
  { 0, { 1, nx, nx + 1 } }
};

TEST(AdjacencyGraphRegularMesh, CheckConnections) {

  const std::string mesh_file = "data/dim2/regular_mesh/database.bin";
  specfem::MPI::MPI *mpi = SPECFEMEnvironment::get_mpi();

  auto mesh =
      specfem::io::read_2d_mesh(mesh_file, specfem::enums::elastic_wave::psv,
                                specfem::enums::electromagnetic_wave::te, mpi);

  const auto &adjacency_graph = mesh.adjacency_graph;
  const auto g = adjacency_graph.graph();

  EXPECT_NO_THROW(adjacency_graph.assert_symmetry());

  // Interior nodes
  for (int ix = 1; ix < nx - 1; ++ix) {
    for (int iz = 1; iz < nz - 1; ++iz) {
      const int ispec = iz * nx + ix;

      // Interior nodes should have 8 neighbors
      const auto neighbors = boost::out_degree(ispec, g);
      EXPECT_EQ(neighbors, 8) << "Interior node at (" << ispec
                              << ") has incorrect number of neighbors";
    }
  }

  // Edge nodes (excluding corners)
  for (int ix = 1; ix < nx - 1; ++ix) {
    {
      const int iz = 0; // Bottom edge
      const int ispec = iz * nx + ix;
      const auto neighbors = boost::out_degree(ispec, g);
      EXPECT_EQ(neighbors, 5) << "Bottom edge node at (" << ispec
                              << ") has incorrect number of neighbors";
    }
    {
      const int iz = nz - 1; // Top edge
      const int ispec = iz * nx + ix;
      const auto neighbors = boost::out_degree(ispec, g);
      EXPECT_EQ(neighbors, 5) << "Top edge node at (" << ispec
                              << ") has incorrect number of neighbors";
    }
  }
  for (int iz = 1; iz < nz - 1; ++iz) {
    {
      const int ix = 0; // Left edge
      const int ispec = iz * nx + ix;
      const auto neighbors = boost::out_degree(ispec, g);
      EXPECT_EQ(neighbors, 5) << "Left edge node at (" << ispec
                              << ") has incorrect number of neighbors";
    }
    {
      const int ix = nx - 1; // Right edge
      const int ispec = iz * nx + ix;
      const auto neighbors = boost::out_degree(ispec, g);
      EXPECT_EQ(neighbors, 5) << "Right edge node at (" << ispec
                              << ") has incorrect number of neighbors";
    }
  }

  // Corner nodes
  {
    const int ispec = 0; // Bottom-left corner
    const auto neighbors = boost::out_degree(ispec, g);
    EXPECT_EQ(neighbors, 3)
        << "Bottom-left corner node has incorrect number of neighbors";
  }
  {
    const int ispec = nx - 1; // Bottom-right corner
    const auto neighbors = boost::out_degree(ispec, g);
    EXPECT_EQ(neighbors, 3)
        << "Bottom-right corner node has incorrect number of neighbors";
  }
  {
    const int ispec = (nz - 1) * nx; // Top-left corner
    const auto neighbors = boost::out_degree(ispec, g);
    EXPECT_EQ(neighbors, 3)
        << "Top-left corner node has incorrect number of neighbors";
  }
  {
    const int ispec = nz * nx - 1; // Top-right corner
    const auto neighbors = boost::out_degree(ispec, g);
    EXPECT_EQ(neighbors, 3)
        << "Top-right corner node has incorrect number of neighbors";
  }

  // Check specific adjacency lists
  for (const auto &entry : expected_adjacency) {
    const int ispec = entry.first;
    const auto expected_neighbors = entry.second;
    std::vector<int> actual_neighbors;

    for (const auto &edge :
         boost::make_iterator_range(boost::out_edges(ispec, g))) {
      const auto target = boost::target(edge, g);
      actual_neighbors.push_back(target);
    }

    std::sort(actual_neighbors.begin(), actual_neighbors.end());
    auto expected_sorted = expected_neighbors;
    std::sort(expected_sorted.begin(), expected_sorted.end());

    std::ostringstream message;
    message << "Adjacency list mismatch for node " << ispec << ":\n"
            << "Expected: ";
    for (const auto &n : expected_sorted) {
      message << n << " ";
    }
    message << "\nActual: ";
    for (const auto &n : actual_neighbors) {
      message << n << " ";
    }
    message << "\n";

    EXPECT_EQ(actual_neighbors, expected_sorted) << message.str();
  }
}
