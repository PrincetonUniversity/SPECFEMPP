#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/compare_array.h"
#include "compute/interface.hpp"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename, rho_file, kappa_file, mu_file, rho_vs_file,
      rho_vp_file, qkappa_file, qmu_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.rho_file = Node["rho_file"].as<std::string>();
  test_config.kappa_file = Node["kappa_file"].as<std::string>();
  test_config.mu_file = Node["mu_file"].as<std::string>();
  test_config.rho_vs_file = Node["rho_vs_file"].as<std::string>();
  test_config.rho_vp_file = Node["rho_vp_file"].as<std::string>();
  test_config.qkappa_file = Node["qkappa_file"].as<std::string>();
  test_config.qmu_file = Node["qmu_file"].as<std::string>();

  return;
}

test_config get_test_config(std::string config_filename,
                            specfem::MPI::MPI *mpi) {
  // read test config file
  YAML::Node yaml = YAML::LoadFile(config_filename);
  test_config test_config{};
  if (mpi->get_size() == 1) {
    YAML::Node Node = yaml["SerialTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == 1);
    for (auto N : database)
      N >> test_config;
  } else {
    YAML::Node Node = yaml["ParallelTest"];
    YAML::Node database = Node["database"];
    assert(database.IsSequence());
    assert(database.size() == Node["config"]["nproc"].as<int>());
    assert(mpi->get_size() == Node["config"]["nproc"].as<int>());
    for (auto N : database) {
      if (N["processor"].as<int>() == mpi->get_rank())
        N >> test_config;
    }
  }

  return test_config;
}
// ---------------------------------------------------------------------------

/**
 *
 * This test should be run on single and multiple nodes
 *
 */
TEST(COMPUTE_TESTS, compute_elastic_properties) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename =
      "../../../tests/unit-tests/compute/elastic/test_config.yml";
  test_config test_config = get_test_config(config_filename, mpi);

  // Set up GLL quadrature points
  specfem::quadrature::quadrature *gllx =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);
  specfem::quadrature::quadrature *gllz =
      new specfem::quadrature::gll::gll(0.0, 0.0, 5);
  std::vector<std::shared_ptr<specfem::material::material> > materials;

  specfem::mesh::mesh mesh(test_config.database_filename, materials, mpi);

  specfem::compute::properties properties(mesh.material_ind.kmato, materials,
                                          mesh.nspec, gllz->get_N(),
                                          gllx->get_N());

  specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight> h_rho =
      properties.h_rho;
  EXPECT_NO_THROW(specfem::testing::test_array(
      h_rho, test_config.rho_file, mesh.nspec, gllz->get_N(), gllx->get_N()));

  specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight> h_kappa =
      properties.h_kappa;
  EXPECT_NO_THROW(specfem::testing::test_array(h_kappa, test_config.kappa_file,
                                               mesh.nspec, gllz->get_N(),
                                               gllx->get_N()));

  specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight> h_mu =
      properties.h_mu;
  EXPECT_NO_THROW(specfem::testing::test_array(
      h_mu, test_config.mu_file, mesh.nspec, gllz->get_N(), gllx->get_N()));

  EXPECT_NO_THROW(
      specfem::testing::test_array(properties.rho_vp, test_config.rho_vp_file,
                                   mesh.nspec, gllz->get_N(), gllx->get_N()));

  EXPECT_NO_THROW(
      specfem::testing::test_array(properties.rho_vs, test_config.rho_vs_file,
                                   mesh.nspec, gllz->get_N(), gllx->get_N()));

  EXPECT_NO_THROW(
      specfem::testing::test_array(properties.qkappa, test_config.qkappa_file,
                                   mesh.nspec, gllz->get_N(), gllx->get_N()));

  EXPECT_NO_THROW(specfem::testing::test_array(properties.qmu,
                                               test_config.qmu_file, mesh.nspec,
                                               gllz->get_N(), gllx->get_N()));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
