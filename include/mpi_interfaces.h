#ifndef MPI_INTERFACES_H
#define MPI_INTERFACES_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"
// #include "../include/mpi_interfaces.tpp"

namespace specfem {
namespace interfaces {

struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  specfem::HostView1d<int> my_neighbors, my_nelmnts_neighbors;
  specfem::HostView3d<int> my_interfaces;
  interface(){};
  interface(const int ninterfaces, const int max_interface_size);
  interface(std::ifstream &stream, const specfem::MPI::MPI *mpi);
};
} // namespace interfaces

// namespace compute{
// namespace mpi_interfaces{

// struct mpi_interfaces {
//   specfem::compute::mpi_interfaces::mpi_interface_type<elastic>
//   elastic_interface{}; mpi_interfaces(){}; mpi_interfaces(const int
//   max_interface_size, const int ninterfaces,
//                  const int ngllx);
//   mpi_interfaces()
// }
// }
// }
} // namespace specfem

#endif
