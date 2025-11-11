#ifndef _SPECFEM_MPI_TPP
#define _SPECFEM_MPI_TPP

#include <iostream>
#include <vector>
#include "specfem_mpi.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

template <typename T> void specfem::MPI::MPI::cout(T s, bool root_only) const {
  // Determine if this rank should output based on root_only parameter
  bool should_output = false;

#ifdef MPI_PARALLEL
  if (root_only) {
    // Only rank 0 outputs when root_only=true (precedence rule)
    should_output = (my_rank == 0);
  } else {
    // When root_only=false, all ranks can output
    should_output = true;
  }
#else
  // Non-MPI build: always output
  should_output = true;
#endif

  if (!should_output) return;

  // Output to log file or stdout
  if (logging_enabled_) {
    log_file_ << s << std::endl;
    if (auto_flush_) {
      log_file_.flush();
    }
  } else {
    std::cout << s << std::endl;
  }
}

template <typename T> void specfem::MPI::MPI::print(T s) const {
  this->cout(s, true);
}


#endif
