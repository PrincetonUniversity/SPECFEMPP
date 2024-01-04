#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/compare_array.h"
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

    // Read sources
    //    if start time is not explicitly specified then t0 is determined using
    //    source frequencies and time shift
    auto [sources, t0] =
        specfem::sources::read_sources(sources_file, setup.get_dt(), mpi);

    // Generate compute structs to be used by the solver
    specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                      gllz);
    specfem::compute::partial_derivatives partial_derivatives(
        mesh.coorg, mesh.material_ind.knods, gllx, gllz);
    specfem::compute::properties material_properties(
        mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
        gllz->get_N());
    specfem::compute::coupled_interfaces::coupled_interfaces coupled_interfaces(
        compute.h_ibool, compute.coordinates.coord,
        material_properties.h_ispec_type, mesh.coupled_interfaces);

    // Set up boundary conditions
    specfem::compute::boundaries boundary_conditions(
        mesh.material_ind.kmato, materials, mesh.acfree_surface,
        mesh.abs_boundary);

    // Locate the sources
    for (auto &source : sources)
      source->locate(compute.coordinates.coord, compute.h_ibool,
                     gllx->get_hxi(), gllz->get_hxi(), mesh.nproc, mesh.coorg,
                     mesh.material_ind.knods, mesh.npgeo,
                     material_properties.h_ispec_type, mpi);

    // User output
    for (auto &source : sources) {
      if (mpi->main_proc())
        std::cout << *source << std::endl;
    }

    // Update solver intialization time
    setup.update_t0(-1.0 * t0);

    // Instantiate the solver and timescheme
    auto it = setup.instantiate_solver();

    // User output
    if (mpi->main_proc())
      std::cout << *it << std::endl;

    // Setup solver compute struct
    const type_real xmax = compute.coordinates.xmax;
    const type_real xmin = compute.coordinates.xmin;
    const type_real zmax = compute.coordinates.zmax;
    const type_real zmin = compute.coordinates.zmin;

    specfem::compute::sources compute_sources(sources, gllx, gllz, xmax, xmin,
                                              zmax, zmin, mpi);

    specfem::compute::receivers compute_receivers;

    // Instantiate domain classes

    try {

      const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);
      specfem::enums::element::quadrature::static_quadrature_points<5> qp5;

      specfem::domain::domain<
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          elastic_domain_static(nglob, qp5, &compute, material_properties,
                                partial_derivatives, boundary_conditions,
                                compute_sources, compute_receivers, gllx, gllz);

      specfem::domain::domain<
          specfem::enums::element::medium::acoustic,
          specfem::enums::element::quadrature::static_quadrature_points<5> >
          acoustic_domain_static(nglob, qp5, &compute, material_properties,
                                 partial_derivatives, boundary_conditions,
                                 compute_sources, compute_receivers, gllx,
                                 gllz);

      // Instantiate coupled interfaces
      specfem::coupled_interface::coupled_interface acoustic_elastic_interface(
          acoustic_domain_static, elastic_domain_static, coupled_interfaces,
          qp5, partial_derivatives, compute.ibool, gllx->get_w(),
          gllz->get_w());

      specfem::coupled_interface::coupled_interface elastic_acoustic_interface(
          elastic_domain_static, acoustic_domain_static, coupled_interfaces,
          qp5, partial_derivatives, compute.ibool, gllx->get_w(),
          gllz->get_w());

      specfem::solver::solver *solver = new specfem::solver::time_marching<
          specfem::enums::element::quadrature::static_quadrature_points<5> >(
          acoustic_domain_static, elastic_domain_static,
          acoustic_elastic_interface, elastic_acoustic_interface, it);

      solver->run();

      elastic_domain_static.sync_field(specfem::sync::DeviceToHost);
      acoustic_domain_static.sync_field(specfem::sync::DeviceToHost);

      if (Test.database.elastic_domain_field != "NULL") {
        specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
            field_elastic = elastic_domain_static.get_host_field();

        type_real tolerance = 0.01;

        specfem::testing::compare_norm(field_elastic,
                                       Test.database.elastic_domain_field,
                                       nglob, ndim, tolerance);
      }

      if (Test.database.acoustic_domain_field != "NULL") {
        specfem::kokkos::HostView1d<type_real, Kokkos::LayoutLeft>
            field_acoustic = Kokkos::subview(
                acoustic_domain_static.get_host_field(), Kokkos::ALL(), 0);

        type_real tolerance = 0.01;

        specfem::testing::compare_norm(field_acoustic,
                                       Test.database.acoustic_domain_field,
                                       nglob, tolerance);
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
