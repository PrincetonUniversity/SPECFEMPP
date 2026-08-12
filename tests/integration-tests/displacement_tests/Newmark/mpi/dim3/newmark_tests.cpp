#include "SPECFEM_Environment.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/runtime_configuration.hpp"
#include "specfem/solver.hpp"
#include "specfem/timescheme.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

// Number of MPI processes this executable is launched with. Provided as a
// per-target compile definition by mpi.cmake (one executable per size). The
// fallback keeps editors/clangd happy when the definition is absent.
#ifndef SPECFEM_MPI_TEST_NPROC
#define SPECFEM_MPI_TEST_NPROC 4
#endif

// ------------------------------------- //
// ------- Test configuration ----------- //

// Helper function to load test configuration from directory
struct TestConfig3D {
  std::string name;
  std::string id;
  std::string description;
  int number_of_processors;
  type_real tolerance;
  std::string specfem_config;
  std::string traces;

  static TestConfig3D load_from_directory(const std::string &test_name) {
    TestConfig3D config;

    // Create the test path by concatenating the base path with the test name
    std::string test_path = "displacement_tests/Newmark/mpi/dim3/" + test_name;

    // Load config.yaml from the test directory
    std::string config_file = test_path + "/config.yaml";

    YAML::Node config_node;
    try {
      config_node = YAML::LoadFile(config_file);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to load config file: " + config_file +
                               " - " + e.what());
    }

    config.id = config_node["id"].as<std::string>();
    config.name = config_node["name"].as<std::string>();
    config.description = config_node["description"].as<std::string>();

    // Load configuration
    YAML::Node config_section = config_node["config"];
    config.number_of_processors = config_section["nproc"].as<int>();
    config.tolerance = config_section["tolerance"].as<type_real>();

    // Load database paths and concatenate with test directory path
    YAML::Node databases = config_node["databases"];
    config.specfem_config =
        test_path + "/" + databases["specfem_config"].as<std::string>();
    config.traces = test_path + "/" + databases["traces"].as<std::string>();

    return config;
  }
};

// ------------------------------------- //

// ----- Parse test directories ------------- //

// tests_mpi.yaml maps each test directory name to its MPI process count. A
// given executable is launched at a single size, so it keeps only the cases
// whose core count matches `target_nproc`. Names are sorted for a deterministic
// gtest order.
std::vector<std::string>
parse_3D_test_directories(const std::string &tests_file, int target_nproc) {
  YAML::Node yaml = YAML::LoadFile(tests_file)["tests3d"];

  std::vector<std::string> test_names;

  for (YAML::const_iterator it = yaml.begin(); it != yaml.end(); ++it) {
    if (it->second.as<int>() == target_nproc)
      test_names.push_back(it->first.as<std::string>());
  }

  std::sort(test_names.begin(), test_names.end());

  return test_names;
}

// Parameterized test fixture for the MPI Newmark tests. Named distinctly from
// the serial fixture so the discovered gtest/ctest names carry "MPI" (e.g.
// DisplacementMPITests/NewmarkMPI.3D/"<case>") and never collide with serial.
class NewmarkMPI : public ::testing::TestWithParam<std::string> {
protected:
  void SetUp() override {
    // Skip if the launched MPI size does not satisfy the requested size.
    if (!SPECFEMEnvironment::IsMPISizeValid()) {
      GTEST_SKIP() << SPECFEMEnvironment::GetMPISizeError();
    }

    // Skip ranks that are outside the active communicator (excluded ranks when
    // the launch size exceeds the requested size).
    if (specfem::MPI::communicator() == MPI_COMM_NULL) {
      GTEST_SKIP() << "Rank " << specfem::MPI::get_rank()
                   << " is outside the participating range [0-"
                   << (SPECFEM_MPI_TEST_NPROC - 1) << "].";
    }
  }

  void TearDown() override {
    // Any cleanup needed for each test
  }
};

TEST_P(NewmarkMPI, 3D) {
  const std::string &test_path = GetParam();

  // Load the test configuration from the directory
  TestConfig3D Test = TestConfig3D::load_from_directory(test_path);

  // The per-test process count must match the size this executable was
  // launched with (tests are grouped into tests_mpi<N>.yaml by size).
  EXPECT_EQ(Test.number_of_processors, SPECFEM_MPI_TEST_NPROC)
      << "Test " << Test.name << " declares nproc=" << Test.number_of_processors
      << " but executable runs with " << SPECFEM_MPI_TEST_NPROC
      << " processes.";

  if (specfem::MPI::main_proc()) {
    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;
  }

  const auto parameter_file = Test.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file);

  // get_databases() applies specfem::MPI::format_proc_filename(), so the
  // configured "mesh-database: .../database.bin" resolves to the per-rank
  // ".../database/proc_N.bin" automatically when size > 1.
  const auto database_filename = setup.get_databases();
  const auto &source_entries = setup.get_source_entries();
  const auto stations_node = setup.get_stations();

  // Set up GLL quadrature points
  const auto quadratures = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  auto mesh = specfem::io::read_3d_mesh(database_filename,
                                        setup.get_attenuation_setup());
  const specfem::simulation::type simulation_type =
      setup.get_simulation_type(mesh.materials.has_attenuation());
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0, starttime] =
      specfem::io::read_sources<specfem::element::dimension_tag::dim3>(
          source_entries, nsteps, setup.get_t0(), dt, simulation_type);
  (void)starttime; // unused in test

  for (auto &source : sources) {
    specfem::Logger::info(
        [&](std::ostringstream &oss) { oss << source->print(); });
  }

  setup.update_t0(t0);

  // --------------------------------------------------------------
  //                   Get receivers
  // --------------------------------------------------------------

  // Read receivers from stations file
  auto receivers = specfem::io::read_3d_receivers(stations_node);

  const auto seismogram_types = setup.get_seismogram_types();

  if (receivers.size() == 0) {
    FAIL() << "--------------------------------------------------\n"
           << "\033[0;31m[FAILED]\033[0m Test failed\n"
           << " - Test: " << Test.name << "\n"
           << " - Error: Stations file does not contain any receivers\n"
           << "--------------------------------------------------\n\n"
           << std::endl;
  }

  const int max_sig_step = setup.get_max_seismogram_step();
  const int nstep_between_samples = setup.get_nstep_between_samples();

  specfem::Logger::info("Creating the Assembly...");

  auto start = std::chrono::high_resolution_clock::now();
  specfem::assembly::assembly<specfem::element::dimension_tag::dim3> assembly(
      mesh, quadratures, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_sig_step, nstep_between_samples,
      simulation_type, setup.allocate_boundary_values(),
      setup.instantiate_property_reader());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  specfem::Logger::info([&](std::ostringstream &oss) {
    oss << "Assembly created in " << elapsed.count() << " seconds.";
  });

  // Instantiate the solver and timescheme
  auto it = setup.instantiate_timescheme(assembly.fields, simulation_type);

  // User output
  specfem::Logger::info(
      [&](std::ostringstream &oss) { oss << it->to_string(); });

  std::shared_ptr<specfem::solver::solver> solver = setup.instantiate_solver<5>(
      setup.get_dt(), assembly, it, simulation_type, {});

  solver->run();

  // --------------------------------------------------------------
  //                   Gather seismograms to the main rank
  // --------------------------------------------------------------

  auto seismograms = assembly.receivers;

  seismograms.sync_seismograms();
  // Collective reduction of per-rank seismograms onto the main rank. Every rank
  // MUST call this; only the comparison below is restricted to the main rank.
  seismograms.gather_to_main();

  // Non-main ranks have done their part (located receivers + contributed to the
  // reduction). The comparison against the reference traces runs on rank 0
  // only, where the full set of stations now holds the gathered data.
  if (!specfem::MPI::main_proc()) {
    return;
  }

  // --------------------------------------------------------------
  //                   Compare against reference traces
  // --------------------------------------------------------------

  // An impl function for the seismogram writer used here for generation
  // of the filenames of files written by `xspecfem3D` in Fortran.
  specfem::io::impl::ChannelGenerator channel_generator(dt);

  for (auto station_info : seismograms.stations()) {

    // Get station and network names
    std::string network_name = station_info.network_name;
    std::string station_name = station_info.station_name;

    // Initialize error and computed norm for each all seismogram types
    // that is each station
    type_real error = 0.0;
    type_real computed_norm = 0.0;

    // Loop over all seismogram types for this station to compute the
    // total error and computed norm for a single station
    for (auto seismogram_type : station_info.get_seismogram_types()) {

      // Initialize filenames vector to hold the seismogram filenames
      std::vector<std::string> filenames;

      // Depending on wavefield, and timestep, get the correct filenames
      for (auto &f : channel_generator.get_station_filenames(
               network_name, station_name, "S3", seismogram_type))
        filenames.push_back(Test.traces + "/" + f);

      // Get the number of components for this seismogram type
      const int ncomponents = filenames.size();

      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
          traces("traces", ncomponents, max_sig_step, 2);

      for (int icomp = 0; icomp < ncomponents; icomp++) {
        const auto trace =
            Kokkos::subview(traces, icomp, Kokkos::ALL, Kokkos::ALL);
        specfem::io::seismogram_reader reader(
            filenames[icomp], specfem::enums::seismogram_format::ascii, trace);
        reader.read();
      }

      int count = 0;
      for (auto [time, value] : seismograms.get_seismogram(
               station_name, network_name, seismogram_type)) {
        for (int icomp = 0; icomp < ncomponents; icomp++) {
          const auto read_time = traces(icomp, count, 0);

          if (std::abs(time - read_time) > 1e-3) {
            FAIL() << "--------------------------------------------------\n"
                   << "\033[0;31m[FAILED]\033[0m Test failed\n"
                   << " - Test name: " << Test.name << "\n"
                   << " - Error: Times do not match\n"
                   << " - Network: " << network_name << "\n"
                   << " - Station: " << station_name << "\n"
                   << " - Component: " << icomp << "\n"
                   << " - Expected:  " << time << "\n"
                   << " - Read Time: " << read_time << "\n"
                   << "--------------------------------------------------\n\n"
                   << std::endl;
          }

          const auto computed_value = traces(icomp, count, 1);
          error += std::sqrt((value[icomp] - computed_value) *
                             (value[icomp] - computed_value));
          computed_norm += std::sqrt(computed_value * computed_value);
        }

        count++;
      }
    }

    if (error / computed_norm > Test.tolerance ||
        std::isnan(error / computed_norm)) {
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - Test: " << Test.name << "\n"
             << " - Error: Norm of the error is greater than the tolerance\n"
             << " - Station: " << station_name << "\n"
             << " - Network: " << network_name << "\n"
             << " - Error: " << error << "\n"
             << " - Norm: " << computed_norm << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }

  std::cout << "--------------------------------------------------\n"
            << "\033[0;32m[PASSED]\033[0m Test name: " << Test.name << "\n"
            << "--------------------------------------------------\n\n"
            << std::endl;
}

// Load test directories and create parameterized test instances. All MPI cases
// live in a single tests_mpi.yaml (name -> core count); this executable keeps
// only the cases whose core count matches its launch size
// (SPECFEM_MPI_TEST_NPROC). Reading just this one file at discovery time is
// safe (it is copied to the test output dir POST_BUILD, before per-case data
// exists).
std::vector<std::string> GetTestDirectories() {
  const std::string tests_filename =
      "displacement_tests/Newmark/mpi/dim3/tests_mpi.yaml";
  return parse_3D_test_directories(tests_filename, SPECFEM_MPI_TEST_NPROC);
}

// Instantiate the parameterized test with all configurations
INSTANTIATE_TEST_SUITE_P(DisplacementMPITests, NewmarkMPI,
                         ::testing::ValuesIn(GetTestDirectories()));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(
      new SPECFEMEnvironment(SPECFEM_MPI_TEST_NPROC));
  return RUN_ALL_TESTS();
}
