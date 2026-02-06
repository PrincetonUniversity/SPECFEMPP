#pragma once

namespace specfem {
namespace mesh {
namespace interfaces {

struct interface {
  // Utilities use to compute MPI buffers
  int ninterfaces, max_interface_size;
  Kokkos::View<int *, Kokkos::HostSpace> my_neighbors;
  Kokkos::View<int *, Kokkos::HostSpace> my_nelmnts_neighbors;
  Kokkos::View<int ***, Kokkos::HostSpace> my_interfaces;
  interface() {};
  interface(const int ninterfaces, const int max_interface_size);
  interface(std::ifstream &stream);
  ~interface() = default;
};
} // namespace interfaces
} // namespace mesh
} // namespace specfem
