#ifndef _SPECFEM_MPI_TPP
#define _SPECFEM_MPI_TPP

#include <iostream>
#include <vector>
#include "specfem_mpi.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

template <typename T> void specfem::MPI::MPI::cout(T s, bool root_only) const {
#ifdef MPI_PARALLEL
  if (my_rank == 0 || !root_only) {
    std::cout << s << std::endl;
  }
#else
  std::cout << s << std::endl;
#endif
}

template <typename T> void specfem::MPI::MPI::print(T s) const {
  this->cout(s, true);
}


#endif
