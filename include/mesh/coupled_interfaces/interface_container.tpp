#ifndef _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_
#define _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_

#include "enumerations/specfem_enums.hpp"
#include "fortranio/interface.hpp"
#include "interface_container.hpp"
#include "specfem_mpi/interface.hpp"

template <specfem::enums::element::type medium1,
          specfem::enums::element::type medium2>
specfem::mesh::interface_container<
    medium1, medium2>::interface_container(const int num_interfaces,
                                           std::ifstream &stream,
                                           const specfem::MPI::MPI *mpi)
    : num_interfaces(num_interfaces),
      medium1_index_mapping("medium1_index_mapping", num_interfaces),
      medium2_index_mapping("medium2_index_mapping", num_interfaces) {

  if (!num_interfaces)
    return;

  int medium1_ispec_l, medium2_ispec_l;

  for (int i = 0; i < num_interfaces; i++) {
    specfem::fortran_IO::fortran_read_line(stream, &medium2_ispec_l,
                                           &medium1_ispec_l);
    medium1_index_mapping(i) = medium1_ispec_l - 1;
    medium2_index_mapping(i) = medium2_ispec_l - 1;
  }

  return;
}

template <specfem::enums::element::type medium1,
          specfem::enums::element::type medium2>
template <specfem::enums::element::type medium>
int specfem::mesh::interface_container<
    medium1, medium2>::get_spectral_elem_index(const int interface_index)
    const {
  if constexpr (medium == medium1) {
    return medium1_index_mapping(interface_index);
  } else if constexpr (medium == medium2) {
    return medium2_index_mapping(interface_index);
  } else {
    throw std::runtime_error("Invalid medium type");
  }
}

#endif /* _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_ */
