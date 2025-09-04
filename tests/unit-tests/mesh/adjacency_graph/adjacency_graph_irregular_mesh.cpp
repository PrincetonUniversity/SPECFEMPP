
#include "../../MPI_environment.hpp"
#include "enumerations/mesh_entities.hpp"
#include "io/interface.hpp"
#include "mesh/adjacency_graph/predicate.hpp"
#include "mesh/mesh.hpp"
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "predicate.tpp"

const static std::unordered_map<std::string, std::string> mesh_files = {
  { "Circular mesh", "data/mesh/circular_mesh/database.bin" },
  { "3 Element Nonconforming",
    "data/mesh/3_elem_nonconforming/database.bin" }
};

const static std::unordered_map<std::string, std::vector<predicate::variant> >
    expected_adjacency_rules = {
      { "Circular mesh",
        {
            predicate::connects(37, 39),
            predicate::connects(37, 38),
            predicate::connects(37, 64),
            predicate::connects(37, 63),
            predicate::connects(37, 36),
            predicate::connects(37, 83),
            predicate::connects(37, 2),
            predicate::connects(37, 3),
            predicate::connects(37, 1),
            predicate::number_of_out_edges(37, 9),
        } },
      { "3 Element Nonconforming",
        {
            predicate::connects(0, specfem::mesh_entity::type::top, 1,
                                specfem::mesh_entity::type::bottom)
                .with(specfem::connections::type::nonconforming),
            predicate::connects(0, specfem::mesh_entity::type::top, 2,
                                specfem::mesh_entity::type::bottom)
                .with(specfem::connections::type::nonconforming),
            predicate::connects(1, 2).with(
                specfem::connections::type::strongly_conforming),
            predicate::number_of_out_edges(0, 2),
            predicate::number_of_out_edges(1, 2),
            predicate::number_of_out_edges(2, 2),
        } }
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
  EXPECT_NO_THROW(adjacency_graph.assert_symmetry());

  for (const auto &rule : expected_adjacency_rules.at(mesh_name)) {
    predicate::verify(rule, adjacency_graph);
  }
}

INSTANTIATE_TEST_SUITE_P(AdjacencyGraphIrregularMesh, CheckConnections,
                         ::testing::Values("Circular mesh",
                                           "3 Element Nonconforming"
                                           // Add more mesh names here as needed
                                           ));
