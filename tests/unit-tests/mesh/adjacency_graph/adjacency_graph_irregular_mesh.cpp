
#include "../../MPI_environment.hpp"
#include "enumerations/mesh_entities.hpp"
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

#include "predicate.tpp"

const static std::unordered_map<std::string, std::string> mesh_files = {
  { "Circular mesh", "data/dim2/circular_mesh/database.bin" },
  { "3 Element Nonconforming",
    "data/dim2/3_elem_nonconforming/database.bin" }
};

const static std::unordered_map<
    std::string, std::vector<specfem::testing::predicate::variant> >
    expected_adjacency_rules = {
      { "Circular mesh",
        {
            specfem::testing::predicate::connects(37, 39),
            specfem::testing::predicate::connects(37, 38),
            specfem::testing::predicate::connects(37, 64),
            specfem::testing::predicate::connects(37, 63),
            specfem::testing::predicate::connects(37, 36),
            specfem::testing::predicate::connects(37, 83),
            specfem::testing::predicate::connects(37, 2),
            specfem::testing::predicate::connects(37, 3),
            specfem::testing::predicate::connects(37, 1),
            specfem::testing::predicate::number_of_out_edges(37, 9),
        } },
      { "3 Element Nonconforming",
        {
            specfem::testing::predicate::connects(
                0, specfem::mesh_entity::type::top, 1,
                specfem::mesh_entity::type::bottom)
                .with(specfem::connections::type::nonconforming),
            specfem::testing::predicate::connects(
                0, specfem::mesh_entity::type::top, 2,
                specfem::mesh_entity::type::bottom)
                .with(specfem::connections::type::nonconforming),
            specfem::testing::predicate::connects(1, 2).with(
                specfem::connections::type::strongly_conforming),
            specfem::testing::predicate::number_of_out_edges(0, 2),
            specfem::testing::predicate::number_of_out_edges(1, 2),
            specfem::testing::predicate::number_of_out_edges(2, 2),
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
    specfem::testing::predicate::verify(rule, adjacency_graph);
  }
}

INSTANTIATE_TEST_SUITE_P(AdjacencyGraphIrregularMesh, CheckConnections,
                         ::testing::Values("Circular mesh",
                                           "3 Element Nonconforming"
                                           // Add more mesh names here as needed
                                           ));
