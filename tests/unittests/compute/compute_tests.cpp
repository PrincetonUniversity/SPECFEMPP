#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "../utilities/include/compare_array.h"
#include "compute/interface.hpp"
#include "material.h"
#include "mesh.h"
#include "quadrature.h"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename, ibool_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.ibool_file = Node["ibool_file"].as<std::string>();

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
 * This test should be run on single and multiple nodes
 *
 */
TEST(COMPUTE_TESTS, compute_ibool) {

  std::cout << "Hello -2" << std::endl;
  std::string config_filename =
      "../../../tests/unittests/compute/test_config.yml";
  test_config test_config =
      get_test_config(config_filename, MPIEnvironment::mpi_);

  // Set up GLL quadrature points
  specfem::quadrature::quadrature gllx(0.0, 0.0, 5);
  specfem::quadrature::quadrature gllz(0.0, 0.0, 5);
  std::vector<specfem::material *> materials;

  specfem::mesh mesh(test_config.database_filename, materials,
                     MPIEnvironment::mpi_);

  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);

  specfem::kokkos::HostView3d<int> h_ibool = compute.h_ibool;
  EXPECT_NO_THROW(specfem::testing::test_array(
      h_ibool, test_config.ibool_file, mesh.nspec, gllz.get_N(), gllx.get_N()));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
