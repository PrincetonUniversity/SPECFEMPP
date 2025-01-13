#include "MPI_environment.hpp"
#include "specfem_mpi/interface.hpp"
#include <memory>

char ***argv = nullptr;
int argc = 0;
std::shared_ptr<specfem::MPI::MPI> MPIEnvironment::mpi_ =
    std::make_shared<specfem::MPI::MPI>(&argc, argv);

void MPIEnvironment::SetUp() {}

void MPIEnvironment::TearDown() {}
