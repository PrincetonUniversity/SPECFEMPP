#pragma once

#include "specfem/mpi.hpp"
#include <vector>

namespace specfem::assembly::receivers_impl {

/**
 * @brief Share every receiver's medium code with all ranks.
 *
 * The owning rank contributes the medium code and every other rank -1; after
 * the reduction every rank holds every receiver's medium.
 *
 * @param medium_codes Per-receiver medium codes, reduced in place
 */
inline void broadcast_medium_tags(std::vector<int> &medium_codes) {
#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(
      MPI_IN_PLACE, medium_codes.data(), static_cast<int>(medium_codes.size()),
      MPI_INT, MPI_MAX, specfem::MPI::communicator()));
#endif
}

} // namespace specfem::assembly::receivers_impl
