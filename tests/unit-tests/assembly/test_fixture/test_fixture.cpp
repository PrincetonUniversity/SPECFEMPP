#include "test_fixture.hpp"
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

  std::string config_filename =
      "../../../tests/unit-tests/assembly/test_config.yaml";
  parse_test_config(YAML::LoadFile(config_filename), Tests);

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  for (auto &Test : Tests) {
    const auto [database_file, sources_file, stations_file] =
        Test.get_databases();
    specfem::mesh::mesh mesh(database_file, mpi);

    const auto [sources, t0] = specfem::sources::read_sources(
        sources_file, 0, 0, 0, specfem::simulation::type::forward);

    const auto receivers = specfem::receivers::read_receivers(stations_file, 0);

    std::vector<specfem::enums::seismogram::type> seismogram_types = {
      specfem::enums::seismogram::type::displacement
    };

    assemblies.push_back(specfem::compute::assembly(
        mesh, quadrature, sources, receivers, seismogram_types, t0, 0, 0, 0,
        specfem::simulation::type::forward));
  }
}

// Instantiate template functions

template KOKKOS_FUNCTION
    specfem::point::index<specfem::dimension::type::dim2, true>
    get_index<true>(const int ielement, const int num_elements, const int iz,
                    const int ix);

template KOKKOS_FUNCTION
    specfem::point::index<specfem::dimension::type::dim2, false>
    get_index<false>(const int ielement, const int num_elements, const int iz,
                     const int ix);
