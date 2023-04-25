#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "material.h"
#include "mesh/mesh.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <string>

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  return;
}

test_config get_test_config(std::string config_filename,
                            specfem::MPI::MPI *mpi) {
  // read test config file
  YAML::Node yaml = YAML::LoadFile(config_filename);
  test_config test_config{};
  if (mpi->get_size() == 1) {
    YAML::Node Node = yaml["SerialTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == 1);
    for (auto N : database)
      N >> test_config;
  } else {
    YAML::Node Node = yaml["ParallelTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == Node["config"]["nproc"].as<int>());
    assert(mpi->get_size() == Node["config"]["nproc"].as<int>());
    for (auto N : database) {
      if (N["processor"].as<int>() == mpi->get_rank())
        N >> test_config;
    }
  }

  return test_config;
}
// ---------------------------------------------------------------------------

/**
 *
 * Check if we can read fortran binary files correctly.
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(MESH_TESTS, fortran_binary_reader) {

  std::string config_filename =
      "../../../tests/unittests/mesh/test_config.yaml";
  test_config test_config =
      get_test_config(config_filename, MPIEnvironment::mpi_);

  std::vector<specfem::material *> materials;
  EXPECT_NO_THROW(specfem::mesh::mesh mesh(test_config.database_filename,
                                           materials, MPIEnvironment::mpi_));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
