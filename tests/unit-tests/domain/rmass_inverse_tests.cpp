#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "../utilities/include/interface.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "receiver/interface.hpp"
#include "source/interface.hpp"
#include "yaml-cpp/yaml.h"

// ------------------------------------- //
// ------- Test configuration ----------- //

namespace test_config {
struct database {
public:
  database()
      : specfem_config(""), elastic_mass_matrix("NULL"),
        acoustic_mass_matrix("NULL"){};
  database(const YAML::Node &Node) {
    specfem_config = Node["specfem_config"].as<std::string>();
    // check if node elastic_mass_matrix exists
    if (Node["elastic_mass_matrix"])
      elastic_mass_matrix = Node["elastic_mass_matrix"].as<std::string>();

    // check if node acoustic_mass_matrix exists
    if (Node["acoustic_mass_matrix"])
      acoustic_mass_matrix = Node["acoustic_mass_matrix"].as<std::string>();
  }
  std::string specfem_config;
  std::string elastic_mass_matrix = "NULL";
  std::string acoustic_mass_matrix = "NULL";
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

TEST(DOMAIN_TESTS, rmass_inverse) {
  std::string config_filename =
      "../../../tests/unit-tests/domain/test_config.yaml";

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
    auto quadratures = setup.instantiate_quadrature();

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh(database_file, mpi);

    // Setup dummy sources and receivers for testing
    std::vector<std::shared_ptr<specfem::sources::source> > sources(0);
    std::vector<std::shared_ptr<specfem::receivers::receiver> > receivers(0);
    std::vector<specfem::enums::seismogram::type> stypes(0);

    // Generate compute structs to be used by the solver
    specfem::compute::assembly assembly(mesh, quadratures, sources, receivers,
                                        stypes, 0, 0,
                                        setup.get_simulation_type());

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

      elastic_domain_static.template mass_time_contribution<
          specfem::enums::time_scheme::type::newmark>(setup.get_dt());
      acoustic_domain_static.template mass_time_contribution<
          specfem::enums::time_scheme::type::newmark>(setup.get_dt());

      elastic_domain_static.invert_mass_matrix();
      acoustic_domain_static.invert_mass_matrix();

      assembly.fields.sync_fields<specfem::sync::DeviceToHost>();

      const int nglob = assembly.fields.forward.nglob;

      // elastic_domain_static.sync_rmass_inverse(specfem::sync::DeviceToHost);
      // acoustic_domain_static.sync_rmass_inverse(specfem::sync::DeviceToHost);

      if (Test.database.elastic_mass_matrix != "NULL") {
        specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
            h_mass_inverse = assembly.fields.forward.elastic.h_mass_inverse;

        specfem::testing::array2d<type_real, Kokkos::LayoutLeft> mass_inverse(
            h_mass_inverse);

        specfem::testing::array2d<type_real, Kokkos::LayoutLeft>
            h_mass_matrix_global(Test.database.elastic_mass_matrix, nglob, 2);

        auto index_mapping = Kokkos::subview(
            assembly.fields.forward.h_assembly_index_mapping, Kokkos::ALL(),
            static_cast<int>(specfem::element::medium_tag::elastic));

        auto h_mass_matrix_local =
            compact_array<specfem::element::medium_tag::elastic>(
                h_mass_matrix_global, index_mapping);

        type_real tolerance = 1e-5;

        ASSERT_TRUE(specfem::testing::compare_norm(
            mass_inverse, h_mass_matrix_local, tolerance));

        // specfem::testing::compare_norm(h_rmass_inverse_static,
        //                                Test.database.elastic_mass_matrix,
        //                                nglob, ndim, tolerance);
      }

      if (Test.database.acoustic_mass_matrix != "NULL") {
        specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
            h_rmass_inverse =
                Kokkos::subview(assembly.fields.forward.acoustic.h_mass_inverse,
                                Kokkos::ALL(), 0);

        specfem::testing::array1d<type_real, Kokkos::LayoutLeft> mass_inverse(
            h_rmass_inverse);

        specfem::testing::array1d<type_real, Kokkos::LayoutLeft>
            h_mass_matrix_global(Test.database.acoustic_mass_matrix, nglob);

        auto index_mapping = Kokkos::subview(
            assembly.fields.forward.h_assembly_index_mapping, Kokkos::ALL(),
            static_cast<int>(specfem::element::medium_tag::acoustic));

        auto h_mass_matrix_local =
            compact_array<specfem::element::medium_tag::acoustic>(
                h_mass_matrix_global, index_mapping);

        type_real tolerance = 1e-5;

        ASSERT_TRUE(specfem::testing::compare_norm(
            mass_inverse, h_mass_matrix_local, tolerance));

        // specfem::testing::compare_norm(h_rmass_inverse_static,
        //                                Test.database.acoustic_mass_matrix,
        //                                nglob, tolerance);
      }

      std::cout << "--------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m Test: " << Test.name << "\n"
                << "--------------------------------------------------\n\n"
                << std::endl;

    } catch (const std::exception &e) {
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

  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
