#include "../../../Kokkos_Environment.hpp"
#include "../../../MPI_environment.hpp"
#include "../../../utilities/include/compare_array.h"
#include "IO/interface.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "medium/material.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "solver/interface.hpp"
#include "timescheme/interface.hpp"
#include "yaml-cpp/yaml.h"

// ----- Parse test config ------------- //

struct test_config {
  std::string specfem_config, solutions_file;
};

void operator>>(const YAML::Node &Node, test_config &test_config) {
  test_config.specfem_config = Node["specfem_config"].as<std::string>();
  test_config.solutions_file = Node["solutions_file"].as<std::string>();

  return;
}

test_config parse_test_config(std::string test_configuration_file,
                              specfem::MPI::MPI *mpi) {

  YAML::Node yaml = YAML::LoadFile(test_configuration_file);
  const YAML::Node &tests = yaml["Tests"];
  const YAML::Node &serial = tests["serial"];

  test_config test_config;
  if (mpi->get_size() == 1) {
    serial >> test_config;
  }

  return test_config;
}

// ------------------------------------- //

TEST(DISPLACEMENT_TESTS, newmark_scheme_tests) {
  std::string config_filename = "displacement_tests/"
                                "Newmark/elastic/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto database_file = setup.get_databases();
  const auto source_node = setup.get_sources();
  // mpi->cout(setup.print_header());

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  std::vector<specfem::medium::material *> materials;
  specfem::mesh::mesh mesh = specfem::IO::read_2d_mesh(database_file, mpi);

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] =
      specfem::IO::read_sources(source_node, setup.get_dt(), mpi);

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

  // Locate the sources
  for (auto &source : sources)
    source->locate(compute.coordinates.coord, compute.h_ibool, gllx->get_hxi(),
                   gllz->get_hxi(), mesh.nproc, mesh.coorg,
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
  const int nglob = specfem::utilities::compute_nglob(compute.h_ibool);
  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  specfem::domain::domain<
      specfem::enums::element::medium::acoustic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      acoustic_domain_static(nglob, qp5, &compute, material_properties,
                             partial_derivatives, compute_sources,
                             compute_receivers, gllx, gllz);
  specfem::domain::domain<
      specfem::enums::element::medium::elastic,
      specfem::enums::element::quadrature::static_quadrature_points<5> >
      elastic_domain_static(nglob, qp5, &compute, material_properties,
                            partial_derivatives, compute_sources,
                            compute_receivers, gllx, gllz);

  // Instantiate coupled interfaces
  specfem::coupled_interface::coupled_interface acoustic_elastic_interface(
      acoustic_domain_static, elastic_domain_static, coupled_interfaces, qp5,
      partial_derivatives, compute.h_ibool, gllx->get_xi(), gllz->get_xi());

  specfem::coupled_interface::coupled_interface elastic_acoustic_interface(
      elastic_domain_static, acoustic_domain_static, coupled_interfaces, qp5,
      partial_derivatives, compute.h_ibool, gllx->get_xi(), gllz->get_xi());

  specfem::solver::solver *solver = new specfem::solver::time_marching<
      specfem::enums::element::quadrature::static_quadrature_points<5> >(
      acoustic_domain_static, elastic_domain_static, acoustic_elastic_interface,
      elastic_acoustic_interface, it);

  solver->run();

  elastic_domain_static.sync_field(specfem::sync::DeviceToHost);

  specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft> field =
      elastic_domain_static.get_host_field();

  type_real tolerance = 0.01;

  EXPECT_NO_THROW(specfem::testing::compare_norm(
      field, test_config.solutions_file, nglob, ndim, tolerance));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
