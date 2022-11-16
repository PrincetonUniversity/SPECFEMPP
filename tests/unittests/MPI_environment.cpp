#include "MPI_environment.hpp"
#include "../../include/specfem_mpi.h"

char **argv;
int argc = 0;
specfem::MPI *MPIEnvironment::mpi_ = new specfem::MPI(&argc, &argv);

void MPIEnvironment::SetUp() {}

void MPIEnvironment::TearDown() { delete MPIEnvironment::mpi_; }
