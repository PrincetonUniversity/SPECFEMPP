#include "MPI_environment.hpp"
#include "specfem_mpi/interface.hpp"

char **argv;
int argc = 0;
specfem::MPI::MPI *MPIEnvironment::mpi_ = new specfem::MPI::MPI(&argc, &argv);

void MPIEnvironment::SetUp() {}

void MPIEnvironment::TearDown() { delete MPIEnvironment::mpi_; }
