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

    setup.update_t0(t0);

    // Instantiate the solver and timescheme
    auto it = setup.instantiate_solver();

    std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers(0);
    std::vector<specfem::enums::seismogram::type> seismogram_types(0);

    const type_real nsteps = it->get_max_timestep();
    specfem::compute::assembly assembly(mesh, quadratures, sources, receivers,
                                        seismogram_types, nsteps, 0,
                                        setup.get_simulation_type());

    // User output
    if (mpi->main_proc())
      std::cout << *it << std::endl;

    // Instantiate domain classes

    try {

      specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

      specfem::domain::domain<
          specfem::simulation::type::forward, specfem::dimension::type::dim2,
          specfem::element::medium_tag::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          elastic_domain_static(assembly, qp5);

      specfem::domain::domain<
          specfem::simulation::type::forward, specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          acoustic_domain_static(assembly, qp5);

      // Instantiate coupled interfaces
      specfem::coupled_interface::coupled_interface<
          specfem::dimension::type::dim2,
          specfem::element::medium_tag::acoustic,
          specfem::element::medium_tag::elastic>
          acoustic_elastic_interface(assembly);

      specfem::coupled_interface::coupled_interface<
          specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
          specfem::element::medium_tag::acoustic>
          elastic_acoustic_interface(assembly);

      std::shared_ptr<specfem::solver::solver> solver = std::make_shared<
          specfem::solver::time_marching<specfem::enums::element::quadrature::
                                             static_quadrature_points<5> > >(
          assembly, acoustic_domain_static, elastic_domain_static,
          acoustic_elastic_interface, elastic_acoustic_interface, it);

      solver->run();

      assembly.fields.sync_fields<specfem::sync::kind::DeviceToHost>();

      const int nglob = assembly.fields.forward.nglob;

      if (Test.database.elastic_domain_field != "NULL") {

        specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
            h_elastic_field = assembly.fields.forward.elastic.h_field;

        specfem::testing::array2d<type_real, Kokkos::LayoutLeft> displacement(
            h_elastic_field);

        specfem::testing::array2d<type_real, Kokkos::LayoutLeft>
            displacement_global(Test.database.elastic_domain_field, nglob, 2);

        auto index_mapping = Kokkos::subview(
            assembly.fields.forward.h_assembly_index_mapping, Kokkos::ALL(),
            static_cast<int>(specfem::element::medium_tag::elastic));

        auto displacement_ref =
            compact_array<specfem::element::medium_tag::elastic>(
                displacement_global, index_mapping);

        type_real tolerance = 0.01;

        ASSERT_TRUE(specfem::testing::compare_norm(
            displacement, displacement_ref, tolerance));
      }

      if (Test.database.acoustic_domain_field != "NULL") {
        specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
            h_acoustic_field = Kokkos::subview(
                assembly.fields.forward.acoustic.h_field, Kokkos::ALL(), 0);

        specfem::testing::array1d<type_real, Kokkos::LayoutLeft> potential(
            h_acoustic_field);

        specfem::testing::array1d<type_real, Kokkos::LayoutLeft>
            potential_global(Test.database.acoustic_domain_field, nglob);

        auto index_mapping = Kokkos::subview(
            assembly.fields.forward.h_assembly_index_mapping, Kokkos::ALL(),
            static_cast<int>(specfem::element::medium_tag::acoustic));

        auto potential_ref =
            compact_array<specfem::element::medium_tag::acoustic>(
                potential_global, index_mapping);

        type_real tolerance = 0.01;

        ASSERT_TRUE(specfem::testing::compare_norm(potential, potential_ref,
                                                   tolerance));
      }

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
