#include "test_fixture.hpp"
#include "IO/interface.hpp"
#include "test_fixture.tpp"

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

ASSEMBLY::ASSEMBLY() {

  std::string config_filename = "assembly/test_config.yaml";
  parse_test_config(YAML::LoadFile(config_filename), Tests);

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  for (auto &Test : Tests) {
    const auto [database_file, sources_file, stations_file] =
        Test.get_databases();
    const auto mesh = specfem::IO::read_2d_mesh(database_file, mpi);

    this->Meshes.push_back(mesh);
    this->suffixes.push_back(Test.suffix);

    std::cout << sources_file << std::endl;

    const auto [sources, t0] = specfem::IO::read_sources(
        sources_file, 1, 0, 0, specfem::simulation::type::forward);

    this->Sources.push_back(sources);

    const auto receivers = specfem::IO::read_receivers(stations_file, 0);

    this->Stations.push_back(receivers);

    std::vector<specfem::enums::seismogram::type> seismogram_types = {
      specfem::enums::seismogram::type::displacement
    };

    this->assemblies.push_back(specfem::compute::assembly(
        mesh, quadrature, sources, receivers, seismogram_types, 1.0, 0.0, 1, 1,
        1, specfem::simulation::type::forward, nullptr));
  }
}
