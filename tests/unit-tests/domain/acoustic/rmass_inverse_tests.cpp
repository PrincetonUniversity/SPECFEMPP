#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/compare_array.h"
#include "IO/interface.hpp"
#include "compute/interface.hpp"
#include "constants.hpp"
#include "domain/interface.hpp"
#include "medium/material.hpp"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
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

TEST(DOMAIN_TESTS, rmass_inverse_elastic_test) {
  std::string config_filename = "domain/acoustic/test_config.yaml";

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  test_config test_config = parse_test_config(config_filename, mpi);

  const std::string parameter_file = test_config.specfem_config;

  specfem::runtime_configuration::setup setup(parameter_file, __default_file__);

  const auto database_file = setup.get_databases();
  const auto source_node = setup.get_sources();

  // Set up GLL quadrature points
  auto [gllx, gllz] = setup.instantiate_quadrature();

  // Read mesh generated MESHFEM
  std::vector<specfem::medium::material *> materials;
  specfem::mesh::mesh mesh = specfem::IO::read_2d_mesh(database_file, mpi);

  // Read sources
  //    if start time is not explicitly specified then t0 is determined using
  //    source frequencies and time shift
  auto [sources, t0] = specfem::IO::read_sources(source_node, 1e-5, mpi);

  // Generate compute structs to be used by the solver
  specfem::compute::compute compute(mesh.coorg, mesh.material_ind.knods, gllx,
                                    gllz);
  specfem::compute::partial_derivatives partial_derivatives(
      mesh.coorg, mesh.material_ind.knods, gllx, gllz);
  specfem::compute::properties material_properties(
      mesh.material_ind.kmato, materials, mesh.nspec, gllx->get_N(),
      gllz->get_N());

  // Set up boundary conditions
  specfem::compute::boundaries boundary_conditions(
      mesh.material_ind.kmato, materials, mesh.acfree_surface,
      mesh.abs_boundary);

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
  setup.update_t0(t0);

  // Update solver intialization time
  setup.update_t0(t0);

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
                             partial_derivatives, boundary_conditions,
                             compute_sources, compute_receivers, gllx, gllz);

  acoustic_domain_static.invert_mass_matrix();

  acoustic_domain_static.sync_rmass_inverse(specfem::sync::DeviceToHost);

  specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
      h_rmass_inverse_static = acoustic_domain_static.get_host_rmass_inverse();

  EXPECT_NO_THROW(specfem::testing::test_array(
      h_rmass_inverse_static, test_config.solutions_file, nglob, 1));

  // const int ngllx = gllx->get_N();
  // const int ngllz = gllz->get_N();

  // specfem::enums::element::quadrature::dynamic_quadrature_points qp(ngllz,
  // ngllx);

  // specfem::domain::domain<
  //     specfem::enums::element::medium::elastic,
  //     specfem::enums::element::quadrature::dynamic_quadrature_points>
  //     elastic_domain_dynamic(ndim, nglob, qp, &compute, material_properties,
  //                            partial_derivatives, &compute_sources,
  //                            &compute_receivers, gllx, gllz);

  // elastic_domain_dynamic.sync_rmass_inverse(specfem::sync::DeviceToHost);

  // specfem::kokkos::HostView2d<type_real, Kokkos::LayoutLeft>
  //     h_rmass_inverse_dynamic =
  //     elastic_domain_dynamic.get_host_rmass_inverse();

  // EXPECT_NO_THROW(specfem::testing::test_array(
  //     h_rmass_inverse_dynamic, test_config.solutions_file, nglob, ndim));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
