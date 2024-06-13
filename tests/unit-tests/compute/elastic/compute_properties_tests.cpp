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
// Compact global arrays to local arrays

namespace {
template <specfem::element::medium_tag element_type,
          specfem::element::property_tag property>
specfem::testing::array3d<type_real, Kokkos::LayoutRight> compact_global_array(
    const specfem::testing::array3d<type_real, Kokkos::LayoutRight>
        &global_array,
    const specfem::mesh::materials &materials) {
  const int nspec = global_array.n1;
  const int ngllz = global_array.n2;
  const int ngllx = global_array.n3;

  int count = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    auto &material_specification = materials.material_index_mapping(ispec);
    if ((material_specification.type == element_type) &&
        (material_specification.property == property)) {
      count++;
    }
  }

  specfem::testing::array3d<type_real, Kokkos::LayoutRight> local_array(
      count, ngllz, ngllx);

  count = 0;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    auto &material_specification = materials.material_index_mapping(ispec);
    if ((material_specification.type == element_type) &&
        (material_specification.property == property)) {
      for (int igllz = 0; igllz < ngllz; ++igllz) {
        for (int igllx = 0; igllx < ngllx; ++igllx) {
          local_array.data(count, igllz, igllx) =
              global_array.data(ispec, igllz, igllx);
        }
      }
      count++;
    }
  }

  return local_array;
}
} // namespace

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
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);
  specfem::quadrature::quadratures quadratures(gll);

  specfem::mesh::mesh mesh(test_config.database_filename, mpi);

  std::cout << mesh.print() << std::endl;

  const int nspec = mesh.nspec;
  const int ngllz = gll.get_N();
  const int ngllx = gll.get_N();

  specfem::compute::properties compute_properties(nspec, ngllz, ngllx,
                                                  mesh.materials);

  Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace> h_rho =
      compute_properties.elastic_isotropic.h_rho;
  specfem::testing::array3d<type_real, Kokkos::LayoutLeft> rho_array(h_rho);

  specfem::testing::array3d<type_real, Kokkos::LayoutRight> rho_global(
      test_config.rho_file, nspec, ngllz, ngllx);

  auto rho_local =
      compact_global_array<specfem::element::medium_tag::elastic,
                           specfem::element::property_tag::isotropic>(
          rho_global, mesh.materials);

  EXPECT_TRUE(rho_array == rho_local);

  // specfem::kokkos::HostView3d<type_real, Kokkos::LayoutRight> h_mu =
  //     compute_properties.elastic_isotropic.h_mu;
  // specfem::testing::array3d<type_real, Kokkos::LayoutRight> mu_array(h_mu);

  // specfem::testing::array3d<type_real, Kokkos::LayoutRight> mu_global(
  //     test_config.mu_file, nspec, ngllz, ngllx);

  // auto mu_local =
  //     compact_global_array<specfem::element::medium_tag::elastic,
  //                          specfem::element::property_tag::isotropic>(
  //         mu_global, mesh.materials);

  // EXPECT_TRUE(mu_array == mu_local);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
