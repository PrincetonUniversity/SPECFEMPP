#include "test_fixture.hpp"
#include "io/interface.hpp"
#include "test_fixture.tpp"

// ------------------------------------------------------------------------
// Reading test config

template <specfem::dimension::type DimensionType>
void parse_test_config(
    const YAML::Node &yaml,
    std::vector<test_configuration::Test<DimensionType> > &tests,
    const std::string &dimension) {
  YAML::Node all_tests = yaml["Tests"][dimension];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_configuration::Test<DimensionType>(N));

  return;
}

// Template specialization for dim2
template <> Assembly<specfem::dimension::type::dim2>::Assembly() {

  std::string config_filename = "assembly/test_config.yaml";
  parse_test_config<specfem::dimension::type::dim2>(
      YAML::LoadFile(config_filename), Tests, "2D");

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  for (auto &Test : Tests) {
    const auto [database_file, sources_file, stations_file] =
        Test.get_databases();

    const auto elastic_wave = Test.get_elastic_wave();
    const auto electromagnetic_wave = Test.get_electromagnetic_wave();
    const auto mesh = specfem::io::read_2d_mesh(database_file, elastic_wave,
                                                electromagnetic_wave, mpi);

    this->Meshes.push_back(mesh);
    this->suffixes.push_back(Test.suffix);

    std::cout << sources_file << std::endl;

    auto [sources, t0] = specfem::io::read_2d_sources(
        sources_file, 1, 0, 0, specfem::simulation::type::forward);

    this->Sources.push_back(sources);

    const auto receivers = specfem::io::read_2d_receivers(stations_file, 0);

    this->Stations.push_back(receivers);

    std::vector<specfem::wavefield::type> seismogram_types = {
      specfem::wavefield::type::displacement
    };

    this->assemblies.push_back(
        specfem::assembly::assembly<specfem::dimension::type::dim2>(
            mesh, quadrature, sources, receivers, seismogram_types, 1.0, 0.0, 1,
            1, 1, specfem::simulation::type::forward, false, nullptr));
  }
}

// Template specialization for dim3
template <> Assembly<specfem::dimension::type::dim3>::Assembly() {

  std::string config_filename = "assembly/test_config.yaml";
  parse_test_config<specfem::dimension::type::dim3>(
      YAML::LoadFile(config_filename), Tests, "3D");

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  for (auto &Test : Tests) {
    const auto [mesh_parameters_file, mesh_database_file, sources_file] =
        Test.get_databases();

    // For 3D, we need different mesh and source reading functions
    const auto mesh = specfem::io::read_3d_mesh(mesh_parameters_file,
                                                mesh_database_file, mpi);

    this->Meshes.push_back(mesh);
    this->suffixes.push_back(Test.suffix);

    std::cout << sources_file << std::endl;

    auto [sources, t0] = specfem::io::read_3d_sources(
        sources_file, 1, 0, 0, specfem::simulation::type::forward);

    this->Sources.push_back(sources);

    std::cout << "Number of sources: " << sources.size() << std::endl;

    // --------------------------------------------------------------
    //                   Get receivers
    // --------------------------------------------------------------

    auto receivers = specfem::io::read_3d_receivers(Test.database.stations);

    this->Stations.push_back(receivers);

    std::cout << "Number of receivers: " << receivers.size() << std::endl;

    std::vector<specfem::wavefield::type> seismogram_types = {
      specfem::wavefield::type::displacement
    };

    this->assemblies.push_back(
        specfem::assembly::assembly<specfem::dimension::type::dim3>(
            mesh, quadrature, sources, receivers, seismogram_types, 1.0, 0.0, 1,
            1, 1, specfem::simulation::type::forward, false, nullptr));

    std::cout << "Created assembly for " << Test.name << std::endl;
  }
}
