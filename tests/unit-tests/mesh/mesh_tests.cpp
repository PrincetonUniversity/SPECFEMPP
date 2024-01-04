#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "material/interface.hpp"
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
 * Check if we can read fortran binary files correctly.
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(MESH_TESTS, fortran_binary_reader) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
  std::string config_filename =
      "../../../tests/unit-tests/mesh/test_config.yaml";
  std::vector<test_configuration::Test> tests;
  parse_test_config(YAML::LoadFile(config_filename), tests);

  for (auto test : tests) {
    std::vector<std::shared_ptr<specfem::material::material> > materials;
    std::cout << "Executing test: " << test.description << std::endl;
    try {
      specfem::mesh::mesh mesh(
          test.databases.filenames[test.configuration.processors - 1],
          materials, mpi);
      std::cout << " - Test passed\n" << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << " - Error: " << e.what() << std::endl;
      FAIL() << " Test failed\n"
             << " - Test name: " << test.name << "\n"
             << " - Number of MPI processors: " << test.configuration.processors
             << "\n"
             << " - Error: " << e.what() << std::endl;
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
