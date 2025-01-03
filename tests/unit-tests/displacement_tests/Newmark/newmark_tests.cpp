#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "IO/interface.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/domain.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "reader/seismogram.hpp"
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
  std::string config_filename = "../../../tests/unit-tests/displacement_tests/"
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

    const auto [database_file, sources_file] = setup.get_databases();

    // Set up GLL quadrature points
    const auto quadratures = setup.instantiate_quadrature();

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh = specfem::IO::read_mesh(database_file, mpi);
    const type_real dt = setup.get_dt();
    const int nsteps = setup.get_nsteps();

    // Read sources
    //    if start time is not explicitly specified then t0 is determined using
    //    source frequencies and time shift
    auto [sources, t0] = specfem::IO::read_sources(
        sources_file, nsteps, setup.get_t0(), dt, setup.get_simulation_type());

    for (auto &source : sources) {
      if (mpi->main_proc())
        std::cout << source->print() << std::endl;
    }

    setup.update_t0(t0);

    // Instantiate the solver and timescheme
    auto it = setup.instantiate_timescheme();

    const auto stations_filename = setup.get_stations_file();
    const auto angle = setup.get_receiver_angle();
    auto receivers = specfem::IO::read_receivers(stations_filename, angle);

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

    specfem::compute::assembly assembly(mesh, quadratures, sources, receivers,
                                        seismogram_types, t0, setup.get_dt(),
                                        nsteps, it->get_max_seismogram_step(),
                                        setup.get_simulation_type(), nullptr);

    it->link_assembly(assembly);

    // User output
    if (mpi->main_proc())
      std::cout << *it << std::endl;

    specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
    std::shared_ptr<specfem::solver::solver> solver =
        setup.instantiate_solver(setup.get_dt(), assembly, it, qp5, {});

    solver->run();

    // --------------------------------------------------------------
    //                   Write Seismograms
    // --------------------------------------------------------------

    auto seismograms = assembly.receivers;

    seismograms.sync_seismograms();

    // --------------------------------------------------------------

    for (int irec = 0; irec < receivers.size(); ++irec) {
      const auto network_name = receivers[irec]->get_network_name();
      const auto station_name = receivers[irec]->get_station_name();

      for (int itype = 0; itype < seismogram_types.size(); itype++) {

        std::vector<std::string> traces_filename;
        std::string stype_name;
        switch (seismogram_types[itype]) {
        case specfem::enums::seismogram::type::displacement:
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXX.semd");
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXZ.semd");
          stype_name = "displacement";
          break;
        case specfem::enums::seismogram::type::velocity:
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXX.semv");
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXZ.semv");
          stype_name = "velocity";
          break;
        case specfem::enums::seismogram::type::acceleration:
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXX.sema");
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".BXZ.sema");
          stype_name = "acceleration";
          break;
        case specfem::enums::seismogram::type::pressure:
          traces_filename.push_back(Test.database.traces + "/" + station_name +
                                    "." + network_name + ".PRE.semp");
          stype_name = "pressure";
          break;
        default:
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Unknown seismogram type\n"
                 << " - Station: " << station_name << "\n"
                 << " - Network: " << network_name << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
          break;
        }

        type_real error_norm = 0.0;
        type_real compute_norm = 0.0;

        for (int i = 0; i < traces_filename.size(); ++i) {
          Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
              traces("traces", seismograms.h_seismogram.extent(0), 2);
          specfem::reader::seismogram reader(
              traces_filename[i], specfem::enums::seismogram::format::ascii,
              traces);
          reader.read();

          const auto l_seismogram = Kokkos::subview(
              seismograms.h_seismogram, Kokkos::ALL(), itype, irec, i);

          const int nsig_steps = l_seismogram.extent(0);

          for (int isig_step = 0; isig_step < nsig_steps; ++isig_step) {
            const type_real time_t = traces(isig_step, 0);
            const type_real value = traces(isig_step, 1);

            const type_real computed_value = l_seismogram(isig_step);

            error_norm +=
                std::sqrt((value - computed_value) * (value - computed_value));
            compute_norm += std::sqrt(value * value);
          }
        }
        if (error_norm / compute_norm > 1e-3 ||
            std::isnan(error_norm / compute_norm)) {
          FAIL() << "--------------------------------------------------\n"
                 << "\033[0;31m[FAILED]\033[0m Test failed\n"
                 << " - Test name: " << Test.name << "\n"
                 << " - Error: Traces do not match\n"
                 << " - Station: " << station_name << "\n"
                 << " - Network: " << network_name << "\n"
                 << " - Seismogram Type: " << stype_name << "\n"
                 << " - Error value: " << error_norm / compute_norm << "\n"
                 << "--------------------------------------------------\n\n"
                 << std::endl;
        }
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
