
#include <boost/graph/filtered_graph.hpp>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "enumerations/connections.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "mesh/dim2/adjacency_graph/adjacency_graph.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"

TEST_F(MESHES, simplemesh_database_read) {
  const auto &mesh_config = get_mesh_by_name("2 staggered elements");
  const auto &mesh = mesh_config.get_mesh();

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  // mesh built: time to test
  const auto &adjacency_graph = mesh.adjacency_graph;
  ASSERT_FALSE(adjacency_graph.empty()) << "Adjacency graph is empty";
  const auto &g = adjacency_graph.graph();
  EXPECT_NO_THROW(adjacency_graph.assert_symmetry());

  // 1R <-> 2L (NC)
  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::dimension::type::dim2>::EdgeProperties;

  std::map<std::pair<int, int>, std::pair<EdgeProperties, bool> >
      expected_adjacencies = {
        std::make_pair(
            std::make_pair(0, 1),
            std::make_pair(
                EdgeProperties(specfem::connections::type::nonconforming,
                               specfem::mesh_entity::type::right),
                false)),
        std::make_pair(
            std::make_pair(1, 0),
            std::make_pair(
                EdgeProperties(specfem::connections::type::nonconforming,
                               specfem::mesh_entity::type::left),
                false))
      };
  const auto [edges_start, edges_end] = boost::edges(g);
  for (auto edge_it = edges_start; edge_it != edges_end; edge_it++) {
    const int elem_in = boost::source(*edge_it, g);
    const int elem_out = boost::target(*edge_it, g);
    const auto connection = g[*edge_it].connection;
    const auto orientation = g[*edge_it].orientation;

    const auto edge_expect =
        expected_adjacencies.find(std::make_pair(elem_in, elem_out));
    if (edge_expect == expected_adjacencies.end()) {
      std::stringstream msg;
      msg << "Unexpected adjacency found: (" << elem_in << " -> " << elem_out
          << ") -- " << specfem::connections::to_string(connection) << ","
          << specfem::mesh_entity::to_string(orientation);
      FAIL() << msg.str();
    }

    if (edge_expect->second.second) {
      std::stringstream msg;
      msg << "Adjacency (" << elem_in << " -> " << elem_out
          << ") has already been marked true. Are there multiple?";
      FAIL() << msg.str();
    }
    edge_expect->second.second = true;

    // check expected props
    const EdgeProperties &prop = edge_expect->second.first;
    if (prop.connection != connection) {
      std::stringstream msg;
      msg << "Adjacency (" << elem_in << " -> " << elem_out
          << ") expected connection type "
          << specfem::connections::to_string(prop.connection) << ". Found "
          << specfem::connections::to_string(connection);
      FAIL() << msg.str();
    }

    if (prop.orientation != orientation) {
      std::stringstream msg;
      msg << "Adjacency (" << elem_in << " -> " << elem_out
          << ") expected orientation "
          << specfem::mesh_entity::to_string(prop.orientation) << ". Found "
          << specfem::mesh_entity::to_string(orientation);
      FAIL() << msg.str();
    }
  }

  // find any non-hit adjacencies.
  for (auto expect_it = expected_adjacencies.begin();
       expect_it != expected_adjacencies.end(); expect_it++) {
    if (!expect_it->second.second) {
      std::stringstream msg;
      const int elem_in = expect_it->first.first;
      const int elem_out = expect_it->first.second;
      const EdgeProperties &prop = expect_it->second.first;
      msg << "Expected Adjacency not found: (" << elem_in << " -> " << elem_out
          << ") --  " << specfem::connections::to_string(prop.connection) << ","
          << specfem::mesh_entity::to_string(prop.orientation);
      FAIL() << msg.str();
    }
  }
}
