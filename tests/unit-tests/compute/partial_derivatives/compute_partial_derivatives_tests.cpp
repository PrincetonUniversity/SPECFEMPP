#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
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
  std::string database_filename, xix_file, xiz_file, gammax_file, gammaz_file,
      jacobian_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.xix_file = Node["xix_file"].as<std::string>();
  test_config.xiz_file = Node["xiz_file"].as<std::string>();
  test_config.gammax_file = Node["gammax_file"].as<std::string>();
  test_config.gammaz_file = Node["gammaz_file"].as<std::string>();
  test_config.jacobian_file = Node["jacobian_file"].as<std::string>();

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
TEST(COMPUTE_TESTS, compute_partial_derivatives) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename =
      "../../../tests/unit-tests/compute/partial_derivatives/test_config.yml";
  test_config test_config = get_test_config(config_filename, mpi);

  // Set up GLL quadrature points
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  specfem::mesh::mesh mesh(test_config.database_filename, mpi);

  specfem::compute::mesh compute_mesh(mesh.control_nodes, quadratures);
  specfem::compute::partial_derivatives partial_derivatives(compute_mesh);

  const int nspec = compute_mesh.control_nodes.nspec;
  const int ngllz = compute_mesh.quadratures.gll.N;
  const int ngllx = compute_mesh.quadratures.gll.N;

  specfem::kokkos::HostView3d<type_real> h_xix = partial_derivatives.h_xix;
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> xix_array(
      h_xix); // convert to array3d
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> xix_ref(
      test_config.xix_file, nspec, ngllz, ngllx);
  EXPECT_TRUE(xix_array == xix_ref);

  specfem::kokkos::HostView3d<type_real> h_gammax =
      partial_derivatives.h_gammax;
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> gammax_array(
      h_gammax); // convert to array3d
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> gammax_ref(
      test_config.gammax_file, nspec, ngllz, ngllx);
  EXPECT_TRUE(gammax_array == gammax_ref);

  specfem::kokkos::HostView3d<type_real> h_gammaz =
      partial_derivatives.h_gammaz;
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> gammaz_array(
      h_gammaz); // convert to array3d
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> gammaz_ref(
      test_config.gammaz_file, nspec, ngllz, ngllx);
  EXPECT_TRUE(gammaz_array == gammaz_ref);

  specfem::kokkos::HostView3d<type_real> h_jacobian =
      partial_derivatives.h_jacobian;
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> jacobian_array(
      h_jacobian); // convert to array3d
  specfem::testing::array3d<type_real, Kokkos::LayoutRight> jacobian_ref(
      test_config.jacobian_file, nspec, ngllz, ngllx);
  EXPECT_TRUE(jacobian_array == jacobian_ref);

  return;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
