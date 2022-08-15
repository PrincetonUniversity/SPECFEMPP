#include "../../../include/mesh.h"
#include "../../../include/params.h"
#include "../../../include/read_material_properties.h"
#include "../../../include/read_mesh_database.h"
#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include <fstream>
#include <iostream>
#include <string>

using HostView = specfem::HostView2d<type_real>;

TEST(parameter_reader, read_mesh_database_header) {
  std::string filename =
      "../../../tests/unittests/parameter_reader/database.bin";
  std::ifstream stream;

  stream.open(filename);

  specfem::mesh mesh{};
  specfem::parameters params{};

  EXPECT_NO_THROW(
      IO::read_mesh_database_header(stream, mesh, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_coorg_elements(stream, mesh, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(
      IO::read_mesh_database_attenuation(stream, params, MPIEnvironment::mpi_));

  EXPECT_NO_THROW(IO::read_material_properties(stream, mesh.properties.numat,
                                               MPIEnvironment::mpi_));

  stream.close();
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
