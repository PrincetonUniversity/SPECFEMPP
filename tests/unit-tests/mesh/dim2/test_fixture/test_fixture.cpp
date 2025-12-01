#include "test_fixture.hpp"
#include "io/interface.hpp"

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_configuration::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());
  int test_counter = 1;
  for (auto N : all_tests)
    tests.push_back(test_configuration::Test(N, test_counter++));

  return;
}

MESH::MESH() {

  std::string config_filename = "mesh/dim2/test_config.yaml";
  parse_test_config(YAML::LoadFile(config_filename), this->Tests);

  for (auto &Test : this->Tests) {
    const auto [database_file, sources_file, stations_file] =
        Test.get_databases();

    const auto elastic_wave = Test.get_elastic_wave();
    const auto electromagnetic_wave = Test.get_electromagnetic_wave();
    specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(
        database_file, elastic_wave, electromagnetic_wave);

    meshes.push_back(mesh);
  }
}
