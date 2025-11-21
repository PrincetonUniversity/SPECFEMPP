#ifndef _MPI_INTERFACES_HPP
#define _MPI_INTERFACES_HPP

#include "kokkos_abstractions.h"

namespace specfem {
namespace mesh {
namespace interfaces {

struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  specfem::kokkos::HostView1d<int> my_neighbors;
  specfem::kokkos::HostView1d<int> my_nelmnts_neighbors;
  specfem::kokkos::HostView3d<int> my_interfaces;
  interface() {};
  interface(const int ninterfaces, const int max_interface_size);
  interface(std::ifstream &stream);
  ~interface() = default;
};
} // namespace interfaces
} // namespace mesh
} // namespace specfem

#endif
