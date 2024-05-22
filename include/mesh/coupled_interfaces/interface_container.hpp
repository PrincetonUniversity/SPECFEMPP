#ifndef _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_HPP_
#define _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_HPP_

#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
struct interface_container {
  constexpr static auto medium1_type = medium1;
  constexpr static auto medium2_type = medium2;

  interface_container(){};

  interface_container(const int num_interfaces, std::ifstream &stream,
                      const specfem::MPI::MPI *mpi);

  int num_interfaces = 0;
  specfem::kokkos::HostView1d<int> medium1_index_mapping; ///< spectral element
                                                          ///< number for the
                                                          ///< ith edge in
                                                          ///< medium 1

  specfem::kokkos::HostView1d<int> medium2_index_mapping; ///< spectral element
                                                          ///< number for the
                                                          ///< ith element in
                                                          ///< medium 2

  template <specfem::element::medium_tag medium>
  int get_spectral_elem_index(const int interface_index) const;
};
} // namespace mesh
} // namespace specfem

#endif /* _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_HPP_ */