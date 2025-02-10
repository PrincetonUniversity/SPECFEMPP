#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "IO/interface.hpp"
#include "IO/seismogram/reader.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "solver/solver.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"

// ------------------------------------- //
// ------- Test configuration ----------- //

namespace test_config {
struct database {
public:
  database() : specfem_config(""), traces(""){};
  database(const YAML::Node &Node) {
    specfem_config = Node["specfem_config"].as<std::string>();
    // check if node elastic_domain_field exists
    if (Node["traces"]) {
      traces = Node["traces"].as<std::string>();
    } else {
      throw std::runtime_error("Traces not found for the test");
    }
  }
  std::string specfem_config;
  std::string traces;
};

struct configuration {
public:
  configuration() : number_of_processors(0){};
  configuration(const YAML::Node &Node) {
    number_of_processors = Node["nproc"].as<int>();
  }
  int number_of_processors;
};

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node config = Node["config"];
    configuration = test_config::configuration(config);

    YAML::Node database_node = Node["databases"];
    try {
      database = test_config::database(database_node);
    } catch (std::runtime_error &e) {
      throw std::runtime_error("Error in test configuration: " + name + "\n" +
                               e.what());
    }
    return;
  }

  std::string name;
  std::string description;
  test_config::database database;
  test_config::configuration configuration;
};
} // namespace test_config

// ------------------------------------- //

// ----- Parse test config ------------- //

std::vector<test_config::Test> parse_test_config(std::string test_config_file,
                                                 specfem::MPI::MPI *mpi) {
  YAML::Node yaml = YAML::LoadFile(test_config_file);
  const YAML::Node &tests = yaml["Tests"];

  assert(tests.IsSequence());

  std::vector<test_config::Test> test_configurations;
  for (auto N : tests)
    test_configurations.push_back(test_config::Test(N));

  return test_configurations;
}

// ------------------------------------- //

template <specfem::element::medium_tag medium>
specfem::testing::array1d<type_real, Kokkos::LayoutLeft> compact_array(
    const specfem::testing::array1d<type_real, Kokkos::LayoutLeft> global,
    const specfem::kokkos::HostView1d<int, Kokkos::LayoutLeft> index_mapping) {

  const int nglob = index_mapping.extent(0);
  const int n1 = global.n1;

  assert(n1 == nglob);

  int max_global_index = std::numeric_limits<int>::min();

  for (int i = 0; i < nglob; ++i) {
    if (index_mapping(i) != -1) {
      max_global_index = std::max(max_global_index, index_mapping(i));
    }
  }

  specfem::testing::array1d<type_real, Kokkos::LayoutLeft> local_array(
      max_global_index + 1);

  for (int i = 0; i < nglob; ++i) {
    if (index_mapping(i) != -1) {
      local_array.data(index_mapping(i)) = global.data(i);
    }
  }

  return local_array;
}

template <specfem::element::medium_tag medium>
specfem::testing::array2d<type_real, Kokkos::LayoutLeft> compact_array(
    const specfem::testing::array2d<type_real, Kokkos::LayoutLeft> global,
    const specfem::kokkos::HostView1d<int, Kokkos::LayoutLeft> index_mapping) {

  const int nglob = index_mapping.extent(0);
  const int n1 = global.n1;
  const int n2 = global.n2;

  assert(n1 == nglob);

  int max_global_index = std::numeric_limits<int>::min();

  for (int i = 0; i < nglob; ++i) {
    if (index_mapping(i) != -1) {
      max_global_index = std::max(max_global_index, index_mapping(i));
    }
  }

  specfem::testing::array2d<type_real, Kokkos::LayoutLeft> local_array(
      max_global_index + 1, n2);

  for (int i = 0; i < nglob; ++i) {
    if (index_mapping(i) != -1) {
      for (int j = 0; j < n2; ++j) {
        local_array.data(index_mapping(i), j) = global.data(i, j);
      }
    }
  }

  return local_array;
}

TEST(DISPLACEMENT_TESTS, newmark_scheme_tests) {
  std::string config_filename = "displacement_tests/"
                                "Newmark/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto Tests = parse_test_config(config_filename, mpi);

  for (auto &Test : Tests) {
    std::cout << "-------------------------------------------------------\n"
              << "\033[0;32m[RUNNING]\033[0m Test: " << Test.name << "\n"
              << "-------------------------------------------------------\n\n"
              << std::endl;

    const auto parameter_file = Test.database.specfem_config;

    specfem::runtime_configuration::setup setup(parameter_file,
                                                __default_file__);

    const auto database_file = setup.get_databases();
    const auto source_node = setup.get_sources();

    // Set up GLL quadrature points
    const auto quadratures = setup.instantiate_quadrature();

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh = specfem::IO::read_2d_mesh(database_file, mpi);
    const type_real dt = setup.get_dt();
    const int nsteps = setup.get_nsteps();

    // Read sources
    //    if start time is not explicitly specified then t0 is determined using
    //    source frequencies and time shift
    auto [sources, t0] = specfem::IO::read_sources(
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
    auto receivers = specfem::IO::read_receivers(stations_node, angle);

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
             << " - Test name: " << Test.name << "\n"
             << " - Error: Stations file does not contain any receivers\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }

    const int max_sig_step = it->get_max_seismogram_step();

    specfem::compute::assembly assembly(
        mesh, quadratures, sources, receivers, seismogram_types, t0,
        setup.get_dt(), nsteps, max_sig_step, it->get_nstep_between_samples(),
        setup.get_simulation_type(), nullptr);

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

    for (auto [station_name, network_name, seismogram_type] :
         seismograms.get_stations()) {
      std::vector<std::string> filename;
      switch (seismogram_type) {
      case specfem::wavefield::type::displacement:
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXX.semd");
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXZ.semd");
        break;
      case specfem::wavefield::type::velocity:
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXX.semv");
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXZ.semv");
        break;
      case specfem::wavefield::type::acceleration:
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXX.sema");
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.BXZ.sema");
        break;
      case specfem::wavefield::type::pressure:
        filename.push_back(Test.database.traces + "/" + network_name + "." +
                           station_name + ".S2.PRE.semp");
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

      const int ncomponents = filename.size();

      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
          traces("traces", ncomponents, max_sig_step, 2);

      for (int icomp = 0; icomp < ncomponents; icomp++) {
        const auto trace =
            Kokkos::subview(traces, icomp, Kokkos::ALL, Kokkos::ALL);
        specfem::IO::seismogram_reader reader(
            filename[icomp], specfem::enums::seismogram::format::ascii, trace);
        reader.read();
      }

      int count = 0;
      type_real error = 0.0;
      type_real computed_norm = 0.0;
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

      if (error / computed_norm > 1e-3 || std::isnan(error / computed_norm)) {
        FAIL() << "--------------------------------------------------\n"
               << "\033[0;31m[FAILED]\033[0m Test failed\n"
               << " - Test name: " << Test.name << "\n"
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
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
