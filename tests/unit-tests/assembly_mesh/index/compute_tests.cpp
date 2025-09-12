#include "../../Kokkos_Environment.hpp"
#include "../../MPI_environment.hpp"
#include "../../utilities/include/interface.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using HostView1d = specfem::kokkos::HostView1d<int>;
using HostView2d = specfem::kokkos::HostView2d<int>;
using HostView3d = specfem::kokkos::HostView3d<int>;

struct coordinates {
  type_real x = -1.0;
  type_real z = -1.0;
};

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename, ibool_file;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  test_config.ibool_file = Node["ibool_file"].as<std::string>();

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
TEST(ASSEMBLY_MESH, compute_ibool) {

  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  std::string config_filename = "assembly_mesh/index/test_config.yml";
  test_config test_config = get_test_config(config_filename, mpi);

  // Set up GLL quadrature points
  specfem::quadrature::gll::gll gll(0.0, 0.0, 5);

  specfem::quadrature::quadratures quadratures(gll);

  // Read mesh generated MESHFEM
  specfem::mesh::mesh mesh = specfem::io::read_2d_mesh(
      test_config.database_filename, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, mpi);

  // Setup compute structs
  specfem::assembly::mesh<specfem::dimension::type::dim2> compute_mesh(
      mesh.tags, mesh.control_nodes, quadratures,
      mesh.adjacency_graph); // mesh assembly

  const auto h_index_mapping = compute_mesh.h_index_mapping;
  const auto h_coord = compute_mesh.h_coord;

  const int nspec = compute_mesh.nspec;
  const int ngllz = compute_mesh.element_grid.ngllz;
  const int ngllx = compute_mesh.element_grid.ngllx;

  type_real nglob;
  Kokkos::parallel_reduce(
      "specfem::utils::compute_nglob",
      specfem::kokkos::HostMDrange<3>({ 0, 0, 0 }, { nspec, ngllz, ngllx }),
      [=](const int ispec, const int iz, const int ix, type_real &l_nglob) {
        l_nglob = l_nglob > h_index_mapping(ispec, iz, ix)
                      ? l_nglob
                      : h_index_mapping(ispec, iz, ix);
      },
      Kokkos::Max<type_real>(nglob));

  nglob++;

  std::vector<coordinates> coord(nglob);

  for (int ix = 0; ix < ngllx; ++ix) {
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int ispec = 0; ispec < nspec; ++ispec) {
        int index = h_index_mapping(ispec, iz, ix);
        if (coord[index].x == -1.0 && coord[index].z == -1.0) {
          coord[index].x = h_coord(0, ispec, iz, ix);
          coord[index].z = h_coord(1, ispec, iz, ix);
        } else {
          EXPECT_NEAR(coord[index].x, h_coord(0, ispec, iz, ix), 1.0e-6);
          EXPECT_NEAR(coord[index].z, h_coord(1, ispec, iz, ix), 1.0e-6);
        }
      }
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
