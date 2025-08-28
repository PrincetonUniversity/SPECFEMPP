#include "../../MPI_environment.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"

TEST(AdjacencyGraphNonconforming, CheckConnections) {
  const std::string mesh_file = "data/mesh/3_elem_nonconforming/database.bin";
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto mesh =
      specfem::io::read_2d_mesh(mesh_file, specfem::enums::elastic_wave::psv,
                                specfem::enums::electromagnetic_wave::te, mpi);

  const auto &adjacency_graph = mesh.adjacency_graph;
  ASSERT_FALSE(adjacency_graph.empty()) << "Adjacency graph is empty";
  const auto g = adjacency_graph.graph();

  EXPECT_NO_THROW(adjacency_graph.assert_symmetry());

  using EdgeProperties = specfem::mesh::adjacency_graph<
      specfem::dimension::type::dim2>::EdgeProperties;

  // these are the expected nonconforming adjacencies
  std::map<std::pair<int, int>, std::pair<specfem::mesh_entity::type, bool> >
      expected_adjacencies = {
        std::make_pair(std::make_pair(0, 1),
                       std::make_pair(specfem::mesh_entity::type::top, false)),
        std::make_pair(
            std::make_pair(1, 0),
            std::make_pair(specfem::mesh_entity::type::bottom, false)),
        std::make_pair(std::make_pair(0, 2),
                       std::make_pair(specfem::mesh_entity::type::top, false)),
        std::make_pair(
            std::make_pair(2, 0),
            std::make_pair(specfem::mesh_entity::type::bottom, false))
      };
  const auto [edges_start, edges_end] = boost::edges(g);
  for (auto edge_it = edges_start; edge_it != edges_end; edge_it++) {
    const int elem_in = boost::source(*edge_it, g);
    const int elem_out = boost::target(*edge_it, g);
    const auto connection = g[*edge_it].connection;
    const auto orientation = g[*edge_it].orientation;

    if (connection != specfem::connections::type::nonconforming) {
      continue;
    }

    const auto edge_expect =
        expected_adjacencies.find(std::make_pair(elem_in, elem_out));
    if (edge_expect == expected_adjacencies.end()) {
      std::stringstream msg;
      msg << "Unexpected nonconforming adjacency found: (" << elem_in << " -> "
          << elem_out << ") -- "
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
    const specfem::mesh_entity::type &orientation_expect =
        edge_expect->second.first;

    if (orientation_expect != orientation) {
      std::stringstream msg;
      msg << "Adjacency (" << elem_in << " -> " << elem_out
          << ") expected orientation "
          << specfem::mesh_entity::to_string(orientation_expect) << ". Found "
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
      const specfem::mesh_entity::type &orientation_expect =
          expect_it->second.first;
      msg << "Expected Adjacency not found: (" << elem_in << " -> " << elem_out
          << ") --  " << ","
          << specfem::mesh_entity::to_string(orientation_expect);
      FAIL() << msg.str();
    }
  }
}
