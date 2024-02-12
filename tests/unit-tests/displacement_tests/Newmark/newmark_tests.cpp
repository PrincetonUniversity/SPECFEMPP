#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "solver/interface.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"

// ------------------------------------- //
// ------- Test configuration ----------- //

namespace test_config {
struct database {
public:
  database()
      : specfem_config(""), elastic_domain_field("NULL"),
        acoustic_domain_field("NULL"){};
  database(const YAML::Node &Node) {
    specfem_config = Node["specfem_config"].as<std::string>();
    // check if node elastic_domain_field exists
    if (Node["elastic_domain_field"])
      elastic_domain_field = Node["elastic_domain_field"].as<std::string>();

    // check if node acoustic_domain_field exists
    if (Node["acoustic_domain_field"])
      acoustic_domain_field = Node["acoustic_domain_field"].as<std::string>();
  }
  std::string specfem_config;
  std::string elastic_domain_field = "NULL";
  std::string acoustic_domain_field = "NULL";
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
    database = test_config::database(database_node);
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
    specfem::mesh::mesh mesh(database_file, mpi);
    const type_real dt = setup.get_dt();

    // Read sources
    //    if start time is not explicitly specified then t0 is determined using
    //    source frequencies and time shift
    auto [sources, t0] = specfem::sources::read_sources(sources_file, dt);

    for (auto &source : sources) {
      if (mpi->main_proc())
        std::cout << source->print() << std::endl;
    }

    setup.update_t0(-1.0 * t0);

    std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers(0);
    std::vector<specfem::enums::seismogram::type> seismogram_types(0);

    specfem::compute::assembly assembly(mesh, quadratures, sources, receivers,
                                        seismogram_types, 0);

    // Instantiate the solver and timescheme
    auto it = setup.instantiate_solver();

    // User output
    if (mpi->main_proc())
      std::cout << *it << std::endl;

    // Instantiate domain classes

    try {

      specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

      specfem::domain::domain<
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          elastic_domain_static(assembly, qp5);

      specfem::domain::domain<
          specfem::enums::element::medium::acoustic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          acoustic_domain_static(assembly, qp5);

      // Instantiate coupled interfaces
      specfem::coupled_interface::coupled_interface<
          specfem::enums::element::medium::acoustic,
          specfem::enums::element::medium::elastic>
          acoustic_elastic_interface(assembly);

      specfem::coupled_interface::coupled_interface<
          specfem::enums::element::medium::elastic,
          specfem::enums::element::medium::acoustic>
          elastic_acoustic_interface(assembly);

      std::shared_ptr<specfem::solver::solver> solver = std::make_shared<
          specfem::solver::time_marching<specfem::enums::element::quadrature::
                                             static_quadrature_points<5> > >(
          assembly, acoustic_domain_static, elastic_domain_static,
          acoustic_elastic_interface, elastic_acoustic_interface, it);

      solver->run();

      assembly.fields.sync_fields<specfem::sync::kind::DeviceToHost>();

      // if (Test.database.elastic_domain_field != "NULL") {
      //   specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
      //       field_elastic = elastic_domain_static.get_host_field();

      //   type_real tolerance = 0.01;

      //   specfem::testing::compare_norm(field_elastic,
      //                                  Test.database.elastic_domain_field,
      //                                  nglob, ndim, tolerance);
      // }

      // if (Test.database.acoustic_domain_field != "NULL") {
      //   specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
      //       field_acoustic = Kokkos::subview(
      //           acoustic_domain_static.get_host_field(), Kokkos::ALL(), 0);

      //   type_real tolerance = 0.0001;

      //   specfem::testing::compare_norm(field_acoustic,
      //                                  Test.database.acoustic_domain_field,
      //                                  nglob, tolerance);
      // }

      std::cout << "--------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m Test: " << Test.name << "\n"
                << "--------------------------------------------------\n\n"
                << std::endl;

    } catch (std::runtime_error &e) {
      std::cout << " - Error: " << e.what() << std::endl;
      FAIL() << "--------------------------------------------------\n"
             << "\033[0;31m[FAILED]\033[0m Test failed\n"
             << " - Test name: " << Test.name << "\n"
             << " - Number of MPI processors: "
             << Test.configuration.number_of_processors << "\n"
             << " - Error: " << e.what() << "\n"
             << "--------------------------------------------------\n\n"
             << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
