#ifndef _SPECFEM_MPI_TPP
#define _SPECFEM_MPI_TPP

#include <iostream>
#include <vector>
#include "specfem_mpi.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

template <typename T> void specfem::MPI::MPI::cout(T s) const {
#ifdef MPI_PARALLEL
  if (my_rank == 0) {
    std::cout << s << std::endl;
  }
#else
  std::cout << s << std::endl;
#endif
}

#endif
