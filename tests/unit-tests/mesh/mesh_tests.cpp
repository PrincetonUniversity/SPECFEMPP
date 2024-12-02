#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "IO/interface.hpp"
#include "IO/mesh/impl/fortran/read_mesh_database.hpp"
#include "mesh/mesh.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <string>

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {
struct configuration {
public:
  int processors;
};

void operator>>(YAML::Node &Node, configuration &configuration) {
  configuration.processors = Node["nproc"].as<int>();
  return;
}

struct databases {
public:
  databases() : processors(0){};
  databases(const int &nproc) : processors(nproc), filenames(nproc){};

  void append(const YAML::Node &Node) {
    filenames[Node["processor"].as<int>()] = Node["filename"].as<std::string>();
  }
  int processors;
  std::vector<std::string> filenames;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node config = Node["config"];
    config >> configuration;
    YAML::Node database = Node["databases"];

    assert(database.IsSequence());
    assert(database.size() == configuration.processors);

    databases = test_configuration::databases(configuration.processors);

    assert(databases.filenames.size() == configuration.processors);

    for (auto N : database)
      databases.append(N);

    assert(databases.filenames.size() == configuration.processors);

    return;
  }

  std::string name;
  std::string description;
  test_configuration::databases databases;
  test_configuration::configuration configuration;
};
} // namespace test_configuration

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_configuration::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_configuration::Test(N));

  return;
}

// ---------------------------------------------------------------------------

/**
 *
 * Check if we can read fortran binary files correctly. TEST one just reading
 * the header
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(MESH_TESTS, fortran_binary_reader_header) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
  std::string config_filename =
      "../../../tests/unit-tests/mesh/test_config.yaml";
  std::vector<test_configuration::Test> Tests;
  parse_test_config(YAML::LoadFile(config_filename), Tests);
  int nspec, npgeo, nproc;

  for (auto Test : Tests) {
    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
    try {

      std::ifstream stream;
      stream.open(Test.databases.filenames[Test.configuration.processors - 1]);

      auto [nspec, npgeo, nproc] =
          specfem::IO::mesh::impl::fortran::read_mesh_database_header(stream,
                                                                      mpi);
      stream.close();
      std::cout << "nspec = " << nspec << std::endl;
      std::cout << "npgeo = " << npgeo << std::endl;
      std::cout << "nproc = " << nproc << std::endl;

      std::cout << "--------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m Test name: " << Test.name << "\n"
                << "--------------------------------------------------\n\n"
                << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << " - Error: " << e.what() << std::endl;
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - Test name: " << Test.name << "\n"
             << " - Error: " << e.what() << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
  SUCCEED();
  return;
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

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
  std::string config_filename =
      "../../../tests/unit-tests/mesh/test_config.yaml";
  std::vector<test_configuration::Test> Tests;
  parse_test_config(YAML::LoadFile(config_filename), Tests);

  for (auto Test : Tests) {
    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
    try {
      specfem::IO::read_mesh(
          Test.databases.filenames[Test.configuration.processors - 1], mpi);
      std::cout << "--------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m Test name: " << Test.name << "\n"
                << "--------------------------------------------------\n\n"
                << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << " - Error: " << e.what() << std::endl;
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - Test name: " << Test.name << "\n"
             << " - Error: " << e.what() << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
  SUCCEED();
  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
