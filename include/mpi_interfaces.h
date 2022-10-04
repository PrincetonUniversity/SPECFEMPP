#ifndef MPI_INTERFACES_H
#define MPI_INTERFACES_H

#include "../include/kokkos_abstractions.h"

namespace specfem {
namespace interfaces {
struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  specfem::HostView1d<int> my_neighbors, my_nelmnts_neighbors;
  specfem::HostView3d<int> my_interfaces;
  interface(){};
  interface(const int ninterfaces, const int max_interface_size);
};
} // namespace interfaces
} // namespace specfem

#endif
