#ifndef _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_
#define _MESH_COUPLED_INTERFACES_IMPL_INTERFACE_CONTAINER_TPP_

#include "enumerations/specfem_enums.hpp"
#include "io/fortranio/interface.hpp"
#include "interface_container.hpp"
#include "specfem_mpi/interface.hpp"

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::mesh::interface_container<
    specfem::dimension::type::dim2, medium1, medium2>::interface_container(const int num_interfaces)
    : num_interfaces(num_interfaces),
      medium1_index_mapping("specfem::mesh::interface_container::medium1_index_mapping", num_interfaces),
      medium2_index_mapping("specfem::mesh::interface_container::medium2_index_mapping", num_interfaces),
      medium1_edge_type("specfem::mesh::interface_container::medium1_edge_type", num_interfaces),
      medium2_edge_type("specfem::mesh::interface_container::medium2_edge_type", num_interfaces) {
  return;
}

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
template <specfem::element::medium_tag medium>
int specfem::mesh::interface_container<
    specfem::dimension::type::dim2, medium1, medium2>::get_spectral_elem_index(const int interface_index)
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
