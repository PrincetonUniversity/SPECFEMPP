#include "../../../include/material.h"
#include "../../../include/mesh.h"
#include "../../../include/params.h"
#include "../../../include/read_material_properties.h"
#include "../../../include/read_mesh_database.h"
#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "yaml-cpp/yaml.h"
#include <fstream>
#include <iostream>
#include <string>

using HostView = specfem::HostView2d<type_real>;

// ------------------------------------------------------------------------
// Reading test config
struct test_config {
  std::string database_filename;
};

void operator>>(YAML::Node &Node, test_config &test_config) {
  test_config.database_filename = Node["database_file"].as<std::string>();
  return;
}

test_config get_test_config(std::string config_filename, specfem::MPI *mpi) {
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

TEST(MPI_parameter_reader, read_mesh_database_header) {

  std::string config_filename =
      "../../../tests/unittests/parameter_reader/test_config.yaml";
  test_config test_config =
      get_test_config(config_filename, MPIEnvironment::mpi_);

  std::ifstream stream;
  stream.open(test_config.database_filename);
  specfem::mesh mesh{};
  specfem::parameters params{};

  EXPECT_NO_THROW(
      IO::read_mesh_database_header(stream, mesh, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_coorg_elements(stream, mesh, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(
      IO::read_mesh_database_attenuation(stream, params, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(std::vector<specfem::material> materials =
                      IO::read_material_properties(
                          stream, mesh.properties.numat, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(
      IO::read_mesh_database_mato(stream, mesh, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_database_interfaces(stream, mesh.inter,
                                                    MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_absorbing_boundaries(
      stream, mesh.abs_boundary, mesh.properties.nelemabs,
      mesh.properties.nspec, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_database_acoustic_forcing(
      stream, mesh.acforcing_boundary, mesh.properties.nelem_acforcing,
      mesh.properties.nspec, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_database_free_surface(
      stream, mesh.acfree_surface, mesh.properties.nelem_acoustic_surface,
      MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_database_coupled(
      stream, mesh.properties.num_fluid_solid_edges,
      mesh.properties.num_fluid_poro_edges,
      mesh.properties.num_solid_poro_edges, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_mesh_database_tangential(
      stream, mesh.tangential_nodes, mesh.properties.nnodes_tangential_curve));

  EXPECT_NO_THROW(IO::read_mesh_database_axial(
      stream, mesh.axial_nodes, mesh.properties.nelem_on_the_axis, mesh.nspec,
      MPIEnvironment::mpi_));

  stream.close();
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
