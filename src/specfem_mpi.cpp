#include "../include/specfem_mpi.h"
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

specfem::MPI::MPI::MPI(int *argc, char ***argv) {
#ifdef MPI_PARALLEL
  MPI_Init(argc, argv);
  this->comm = MPI_COMM_WORLD;
  MPI_Comm_size(this->comm, &this->world_size);
  MPI_Comm_rank(this->comm, &this->my_rank);
#else
  this->world_size = 1;
  this->my_rank = 0;
#endif
}

void specfem::MPI::MPI::sync_all() const {
#ifdef MPI_PARALLEL
  MPI_Barrier(this->comm);
#endif
}

specfem::MPI::MPI::~MPI() {
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
}

int specfem::MPI::MPI::get_size() const { return this->world_size; }

int specfem::MPI::MPI::get_rank() const { return this->my_rank; }

void specfem::MPI::MPI::exit() {
#ifdef MPI_PARALLEL
  int ierr = MPI_Abort(this->comm, 30);
#else
  std::exit(30);
#endif
}

int specfem::MPI::MPI::reduce(int lvalue,
                              specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  int svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_INT, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

int specfem::MPI::MPI::all_reduce(int lvalue,
                                  specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  int svalue;

  MPI_All_Reduce(&lvalue, &svalue, 1, MPI_INT, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

float specfem::MPI::MPI::reduce(float lvalue,
                                specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  float svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_FLOAT, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

float specfem::MPI::MPI::all_reduce(float lvalue,
                                    specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  float svalue;

  MPI_All_Reduce(&lvalue, &svalue, 1, MPI_FLOAT, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

double specfem::MPI::MPI::reduce(double lvalue,
                                 specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  double svalue;

  MPI_Reduce(&lvalue, &svalue, 1, MPI_DOUBLE, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}

double specfem::MPI::MPI::all_reduce(double lvalue,
                                     specfem::MPI::reduce_type reducer) const {
#ifdef MPI_PARALLEL
  double svalue;

  MPI_All_Reduce(&lvalue, &svalue, 1, MPI_DOUBLE, reducer, 0, this->comm);

  return svalue;
#else
  return lvalue;
#endif
}
