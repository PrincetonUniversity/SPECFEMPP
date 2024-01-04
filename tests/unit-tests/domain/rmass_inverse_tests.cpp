#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "../utilities/include/compare_array.h"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
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

TEST(DOMAIN_TESTS, rmass_inverse) {
  std::string config_filename =
      "../../../tests/unit-tests/domain/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::mpi_;

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
    auto [gllx, gllz] = setup.instantiate_quadrature();

    // Read mesh generated MESHFEM
    std::vector<std::shared_ptr<specfem::material::material> > materials;
    specfem::mesh::mesh mesh(database_file, materials, mpi);

    // Generate compute structs to be used by the solver
    specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                      gllz);
    specfem::compute::partial_derivatives partial_derivatives(
        mesh.coorg, mesh.material_ind.knods, gllx, gllz);
    specfem::compute::properties material_properties(
        mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
        gllz->get_N());
    specfem::compute::boundaries boundary_conditions(
        mesh.material_ind.kmato, materials, mesh.acfree_surface,
        mesh.abs_boundary);

    try {

      const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);
      specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

      specfem::domain::domain<
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          elastic_domain_static(nglob, qp5, &compute, material_properties,
                                partial_derivatives, boundary_conditions,
                                specfem::compute::sources(),
                                specfem::compute::receivers(), gllx, gllz);

      specfem::domain::domain<
          specfem::enums::element::medium::acoustic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          acoustic_domain_static(nglob, qp5, &compute, material_properties,
                                 partial_derivatives, boundary_conditions,
                                 specfem::compute::sources(),
                                 specfem::compute::receivers(), gllx, gllz);

      elastic_domain_static.template mass_time_contribution<
          specfem::enums::time_scheme::type::newmark>(setup.get_dt());
      acoustic_domain_static.template mass_time_contribution<
          specfem::enums::time_scheme::type::newmark>(setup.get_dt());

      elastic_domain_static.invert_mass_matrix();
      acoustic_domain_static.invert_mass_matrix();

      elastic_domain_static.sync_rmass_inverse(specfem::sync::DeviceToHost);
      acoustic_domain_static.sync_rmass_inverse(specfem::sync::DeviceToHost);

      if (Test.database.elastic_mass_matrix != "NULL") {
        specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
            h_rmass_inverse_static =
                elastic_domain_static.get_host_rmass_inverse();

        type_real tolerance = 1e-5;

        specfem::testing::compare_norm(h_rmass_inverse_static,
                                       Test.database.elastic_mass_matrix, nglob,
                                       ndim, tolerance);
      }

      if (Test.database.acoustic_mass_matrix != "NULL") {
        specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
            h_rmass_inverse_static =
                Kokkos::subview(acoustic_domain_static.get_host_rmass_inverse(),
                                Kokkos::ALL(), 0);

        type_real tolerance = 1e-5;

        specfem::testing::compare_norm(h_rmass_inverse_static,
                                       Test.database.acoustic_mass_matrix,
                                       nglob, tolerance);
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
