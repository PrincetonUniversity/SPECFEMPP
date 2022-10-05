#ifndef MPI_INTERFACES_H
#define MPI_INTERFACES_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {
namespace interfaces {
struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  specfem::HostView1d<int> my_neighbors, my_nelmnts_neighbors;
  specfem::HostView3d<int> my_interfaces;
  interface(){};
  interface(const int ninterfaces, const int max_interface_size);
  interface(std::ifstream &stream, const specfem::MPI *mpi);
};
} // namespace interfaces
} // namespace specfem

#endif
