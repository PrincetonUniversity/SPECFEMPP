#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "constants.hpp"
#include "io/interface.hpp"
#include "io/seismogram/reader.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "solver/solver.hpp"
#include "specfem/assembly.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"
#include <algorithm>
#include <boost/filesystem.hpp>

// ------------------------------------- //
// ------- Test configuration ----------- //

// Helper function to load test configuration from directory
struct TestConfig {
  std::string name;
  std::string id;
  std::string description;
  int number_of_processors;
  type_real tolerance;
  std::string specfem_config;
  std::string traces;

  static TestConfig load_from_directory(const std::string &test_name) {
    TestConfig config;

    // Create the test path by concatenating the base path with the test name
    std::string test_path = "displacement_tests/Newmark/serial/" + test_name;

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

std::vector<std::string> parse_test_directories(const std::string &tests_file) {
  YAML::Node yaml = YAML::LoadFile(tests_file);

  std::vector<std::string> test_names;

  for (const auto &test_node : yaml) {
    std::string path = test_node.as<std::string>();
    test_names.push_back(path);
  }

  return test_names;
}

// Parameterized test fixture for Newmark tests
class Newmark : public ::testing::TestWithParam<std::string> {
protected:
  void SetUp() override {
    // Any setup needed for each test
  }

  void TearDown() override {
    // Any cleanup needed for each test
  }
};

TEST_P(Newmark, Test) {
  const std::string &test_path = GetParam();

  // Load the test configuration from the directory
  TestConfig Test = TestConfig::load_from_directory(test_path);

  std::cout << "-------------------------------------------------------\n"
            << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
            << "-------------------------------------------------------\n\n"
            << std::endl;

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto parameter_file = Test.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto database_file = setup.get_databases();
  const auto source_node = setup.get_sources();
  const auto elastic_wave = setup.get_elastic_wave_type();
  const auto electromagnetic_wave = setup.get_electromagnetic_wave_type();

  // Set up GLL quadrature points
  const auto quadratures = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(
      database_file, elastic_wave, electromagnetic_wave, mpi);
  const type_real dt = setup.get_dt();
  const int nsteps = setup.get_nsteps();

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] = specfem::io::read_2d_sources(
      source_node, nsteps, setup.get_t0(), dt, setup.get_simulation_type());

  for (auto &source : sources) {
    if (mpi->main_proc())
      std::cout << source->print() << std::endl;
  }

  setup.update_t0(t0);

  // Instantiate the solver and timescheme
  auto it = setup.instantiate_timescheme();

  const auto stations_node = setup.get_stations();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::io::read_2d_receivers(stations_node, angle);

  std::cout << "  Receiver information\n";
  std::cout << "------------------------------" << std::endl;
  for (auto &receiver : receivers) {
    if (mpi->main_proc())
      std::cout << receiver->print() << std::endl;
  }

  const auto seismogram_types = setup.get_seismogram_types();

  // Check only displacement seismogram types are being computed

  if (receivers.size() == 0) {
    FAIL() << "--------------------------------------------------\n"
           << "\033[0;31m[FAILED]\033[0m Test failed\n"
           << " - Test: " << Test.name << "\n"
           << " - Error: Stations file does not contain any receivers\n"
           << "--------------------------------------------------\n\n"
           << std::endl;
  }

  const int max_sig_step = it->get_max_seismogram_step();

  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
      mesh, quadratures, sources, receivers, seismogram_types, t0,
      setup.get_dt(), nsteps, max_sig_step, it->get_nstep_between_samples(),
      setup.get_simulation_type(), false, nullptr);

  it->link_assembly(assembly);

  // User output
  if (mpi->main_proc())
    std::cout << *it << std::endl;

  std::shared_ptr<specfem::solver::solver> solver =
      setup.instantiate_solver<5>(setup.get_dt(), assembly, it, {});

  solver->run();

  // --------------------------------------------------------------
  //                   Write Seismograms
  // --------------------------------------------------------------

  auto seismograms = assembly.receivers;

  seismograms.sync_seismograms();

  // --------------------------------------------------------------

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

      switch (seismogram_type) {
      case specfem::wavefield::type::displacement:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.semd");
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXX.semd");
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXZ.semd");
        }
        break;
      case specfem::wavefield::type::velocity:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.semv");
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXX.semv");
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXZ.semv");
        }
        break;
      case specfem::wavefield::type::acceleration:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.sema");
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXX.sema");
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXZ.sema");
        }
        break;
      case specfem::wavefield::type::pressure:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Pressure seismograms are not supported for SH "
                    "waves\n"
                 << " - Network: " << network_name << "\n"
                 << " - Station: " << station_name << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.PRE.semp");
        }
        break;
      case specfem::wavefield::type::rotation:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Rotation seismograms are not supported for SH"
                    "waves\n"
                 << " - Network: " << network_name << "\n"
                 << " - Station: " << station_name << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.semr");
        }
        break;
      case specfem::wavefield::type::intrinsic_rotation:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Intrinsic Rotation seismograms "
                    "are not supported for SH waves\n"
                 << " - Network: " << network_name << "\n"
                 << " - Station: " << station_name << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.semir");
        }
        break;
      case specfem::wavefield::type::curl:
        if (elastic_wave == specfem::enums::elastic_wave::sh) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Curl seismograms are not supported for SH"
                    "waves\n"
                 << " - Network: " << network_name << "\n"
                 << " - Station: " << station_name << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        } else if (elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames.push_back(Test.traces + "/" + network_name + "." +
                              station_name + ".S2.BXY.semc");
        }
        break;
      default:
        FAIL() << "--------------------------------------------------\n"
               << "\033[0;31m[FAILED]\033[0m Test failed\n"
               << " - Test name: " << Test.name << "\n"
               << " - Error: Unknown seismogram type\n"
               << " - Network: " << network_name << "\n"
               << " - Station: " << station_name << "\n"
               << "--------------------------------------------------\n\n"
               << std::endl;
        break;
      }

      // Get the number of components for this seismogram type
      const int ncomponents = filenames.size();

      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
          traces("traces", ncomponents, max_sig_step, 2);

      for (int icomp = 0; icomp < ncomponents; icomp++) {
        const auto trace =
            Kokkos::subview(traces, icomp, Kokkos::ALL, Kokkos::ALL);
        specfem::io::seismogram_reader reader(
            filenames[icomp], specfem::enums::seismogram::format::ascii, trace);
        reader.read();
      }

      int count = 0;
      for (auto [time, value] : seismograms.get_seismogram(
               station_name, network_name, seismogram_type)) {
        for (int icomp = 0; icomp < ncomponents; icomp++) {
          const auto computed_time = traces(icomp, count, 0);

          if (std::abs(time - computed_time) > 1e-3) {
            FAIL() << "--------------------------------------------------\n"
                   << "\033[0;31m[FAILED]\033[0m Test failed\n"
                   << " - Test name: " << Test.name << "\n"
                   << " - Error: Times do not match\n"
                   << " - Network: " << network_name << "\n"
                   << " - Station: " << station_name << "\n"
                   << " - Component: " << icomp << "\n"
                   << " - Expected: " << time << "\n"
                   << " - Computed: " << computed_time << "\n"
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
             << " - Error: Norm of the error is greater than 1e-3\n"
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

// Load test directories and create parameterized test instances
std::vector<std::string> GetTestDirectories() {
  std::string tests_filename = "displacement_tests/Newmark/tests.yaml";
  return parse_test_directories(tests_filename);
}

// Instantiate the parameterized test with all configurations
INSTANTIATE_TEST_SUITE_P(DisplacementTests, Newmark,
                         ::testing::ValuesIn(GetTestDirectories()));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
