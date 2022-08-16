#include "../include/specfem_mpi.h"
#include <stdlib.h>

specfem::MPI::MPI(int *argc, char ***argv) {
#ifdef MPI_PARALLEL
  if (specfem::MPI::MPI_Initialized)
    throw std::runtime_error("MPI has already been initialized");
  MPI_Init(argc, argv);
  this->comm = MPI_COMM_WORLD;
  MPI_Comm_size(this->comm, &this->world_size);
  MPI_Comm_rank(this->comm, &this->my_rank);
#else
  this->world_size = 1;
  this->my_rank = 0;
#endif
}

void specfem::MPI::sync_all() {
#ifdef MPI_PARALLEL
  MPI_Barrier(this->comm);
#endif
}

specfem::MPI::~MPI() {
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}

int specfem::MPI::get_size() { return this->world_size; }

int specfem::MPI::get_rank() { return this->my_rank; }

void specfem::MPI::exit() {
#ifdef MPI_PARALLEL
  int ierr = MPI_Abort(this->comm, 30);
#else
  std::exit(30);
#endif
}
