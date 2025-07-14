#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "../utilities/include/interface.hpp"
#include "constants.hpp"
#include "io/interface.hpp"
#include "kokkos_kernels/domain_kernels.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/receivers.hpp"
#include "yaml-cpp/yaml.h"

// ------------------------------------- //
// ------- Test configuration ----------- //

namespace test_config {
struct database {
public:
  database()
      : specfem_config(""), elastic_mass_matrix("NULL"),
        acoustic_mass_matrix("NULL") {};
  database(const YAML::Node &Node) {
    specfem_config = Node["specfem_config"].as<std::string>();
    // check if node elastic_mass_matrix exists
    if (Node["elastic_mass_matrix"])
      elastic_mass_matrix = Node["elastic_mass_matrix"].as<std::string>();

    // check if node acoustic_mass_matrix exists
    if (Node["acoustic_mass_matrix"])
      acoustic_mass_matrix = Node["acoustic_mass_matrix"].as<std::string>();

    if (Node["index_mapping"]) {
      index_mapping = Node["index_mapping"].as<std::string>();
    } else {
      throw std::runtime_error("Index mapping not provided for testing.");
    }
  }
  std::string specfem_config;
  std::string elastic_mass_matrix = "NULL";
  std::string acoustic_mass_matrix = "NULL";
  std::string index_mapping = "NULL";
};

struct configuration {
public:
  configuration() : number_of_processors(0) {};
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

TEST(DOMAIN_TESTS, rmass_inverse) {
  std::string config_filename = "domain/test_config.yaml";

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

    // Set up GLL quadrature points
    auto quadratures = setup.instantiate_quadrature();

    std::cout << "Reading mesh file: " << database_file << std::endl;

    // Read mesh generated MESHFEM
    specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(
        database_file, specfem::enums::elastic_wave::psv,
        specfem::enums::electromagnetic_wave::te, mpi);

    std::cout << "Setting up sources and receivers" << std::endl;

    // Setup dummy sources and receivers for testing
    std::vector<std::shared_ptr<specfem::sources::source> > sources(0);
    std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >
        receivers(0);
    std::vector<specfem::wavefield::type> stypes(0);

    // Generate compute structs to be used by the solver
    specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
        mesh, quadratures, sources, receivers, stypes, 0, 0, 0, 0, 1,
        setup.get_simulation_type(), nullptr);

    try {

      const type_real dt = setup.get_dt();

      specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::forward,
          specfem::dimension::type::dim2, 5>
          kernels(assembly);

      kernels.initialize(dt);
      const auto &elastic_psv_field =
          assembly.fields.forward
              .get_field<specfem::element::medium_tag::elastic_psv>();
      const auto &acoustic_field =
          assembly.fields.forward
              .get_field<specfem::element::medium_tag::acoustic>();

      Kokkos::deep_copy(elastic_psv_field.h_mass_inverse,
                        elastic_psv_field.mass_inverse);

      Kokkos::deep_copy(acoustic_field.h_mass_inverse,
                        acoustic_field.mass_inverse);

      const int nglob = assembly.fields.forward.nglob;

      const int nspec = assembly.mesh.nspec;
      const int ngllz = assembly.mesh.ngllz;
      const int ngllx = assembly.mesh.ngllx;

      const auto global_index_mapping = assembly.mesh.h_index_mapping;

      if ((Test.database.acoustic_mass_matrix == "NULL") &&
          (Test.database.elastic_mass_matrix == "NULL")) {
        throw std::runtime_error(
            "No mass matrix provided for testing. Please provide a mass matrix "
            "for testing.");
      }

      std::cout << "Checking mass matrix inversion for elastic medium " << nglob
                << std::endl;

      if (Test.database.elastic_mass_matrix != "NULL") {
        specfem::testing::array2d<double, Kokkos::LayoutRight>
            h_mass_matrix_global(Test.database.elastic_mass_matrix, nglob, 2);

        specfem::testing::array3d<int, Kokkos::LayoutRight> index_mapping(
            Test.database.index_mapping, nspec, ngllz, ngllx);

        type_real error_norm = 0.0;
        type_real ref_norm = 0.0;

        for (int ix = 0; ix < ngllx; ++ix) {
          for (int iz = 0; iz < ngllz; ++iz) {
            for (int ispec = 0; ispec < nspec; ++ispec) {
              specfem::point::index<specfem::dimension::type::dim2> index(
                  ispec, iz, ix);
              const int ispec_mesh = assembly.mesh.compute_to_mesh(ispec);
              if (assembly.element_types.get_medium_tag(ispec) ==
                  specfem::element::medium_tag::elastic_psv) {

                constexpr int components = 2;
                const auto point_field = [&]() {
                  specfem::point::field<
                      specfem::dimension::type::dim2,
                      specfem::element::medium_tag::elastic_psv, false, false,
                      false, true, false>
                      point_field;
                  specfem::assembly::load_on_host(
                      index, assembly.fields.forward, point_field);
                  return point_field;
                }();

                for (int icomp = 0; icomp < components; ++icomp) {
                  const int iglob = index_mapping.data(ispec_mesh, iz, ix);
                  const auto rmass_ref =
                      h_mass_matrix_global.data(iglob, icomp);
                  const auto rmass = point_field.mass_matrix(icomp);
                  if (std::isnan(rmass)) {
                    std::cout << "rmass: " << rmass << std::endl;
                    std::cout << "rmass_ref: " << rmass_ref << std::endl;
                    std::cout << "ispec: " << ispec << std::endl;
                    std::cout << "iz: " << iz << std::endl;
                    std::cout << "ix: " << ix << std::endl;
                  }
                  error_norm +=
                      std::sqrt((rmass - rmass_ref) * (rmass - rmass_ref));
                  ref_norm += std::sqrt(rmass_ref * rmass_ref);
                }
              }
            }
          }
        }

        type_real tolerance = 1e-5;

        std::cout << "Error norm: " << error_norm << std::endl;
        std::cout << "Ref norm: " << ref_norm << std::endl;

        ASSERT_NEAR(error_norm / ref_norm, 0.0, tolerance);
      }

      if (Test.database.acoustic_mass_matrix != "NULL") {
        specfem::testing::array2d<double, Kokkos::LayoutRight>
            h_mass_matrix_global(Test.database.acoustic_mass_matrix, nglob, 1);

        specfem::testing::array3d<int, Kokkos::LayoutRight> index_mapping(
            Test.database.index_mapping, nspec, ngllz, ngllx);

        type_real error_norm = 0.0;
        type_real ref_norm = 0.0;

        for (int ix = 0; ix < ngllx; ++ix) {
          for (int iz = 0; iz < ngllz; ++iz) {
            for (int ispec = 0; ispec < nspec; ++ispec) {
              specfem::point::index<specfem::dimension::type::dim2> index(
                  ispec, iz, ix);
              const int ispec_mesh = assembly.mesh.compute_to_mesh(ispec);
              if (assembly.element_types.get_medium_tag(ispec) ==
                  specfem::element::medium_tag::acoustic) {

                constexpr int components = 1;
                const auto point_field = [&]() {
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        false, false, false, true, false>
                      point_field;
                  specfem::assembly::load_on_host(
                      index, assembly.fields.forward, point_field);
                  return point_field;
                }();

                for (int icomp = 0; icomp < components; ++icomp) {
                  const int iglob = index_mapping.data(ispec_mesh, iz, ix);
                  const auto rmass_ref =
                      h_mass_matrix_global.data(iglob, icomp);
                  const auto rmass = point_field.mass_matrix(icomp);
                  error_norm +=
                      std::sqrt((rmass - rmass_ref) * (rmass - rmass_ref));
                  ref_norm += std::sqrt(rmass_ref * rmass_ref);
                }
              }
            }
          }
        }

        type_real tolerance = 1e-5;

        std::cout << "Error norm: " << error_norm << std::endl;
        std::cout << "Ref norm: " << ref_norm << std::endl;

        ASSERT_NEAR(error_norm / ref_norm, 0.0, tolerance);
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
