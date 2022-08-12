#include "../../../include/config.h"
#include "../../../include/fortran_IO.h"
#include "../../../include/mesh.h"
#include "../../../include/read_mesh_database.h"
#include "../MPI_environment.hpp"
#include <fstream>
#include <iostream>
#include <string>

TEST(parameter_reader, read_mesh_database_header) {
  std::string filename =
      "../../../tests/unittests/parameter_reader/database.bin";
  std::ifstream stream;

  int dummy_i;
  type_real dummy_d1, dummy_d2;

  stream.open(filename);

  specfem::mesh mesh;
  IO::read_mesh_database_header(stream, mesh, MPIEnvironment::mpi_);

  IO::fortran_IO::fortran_read_line(stream, &dummy_i, &dummy_d1, &dummy_d2);

  EXPECT_EQ(dummy_i, 1);
  EXPECT_FLOAT_EQ(dummy_d1, 0.0);
  EXPECT_FLOAT_EQ(dummy_d2, 0.0);

  stream.close();

  EXPECT_TRUE(true);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::Environment *const mpi_env =
      ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
