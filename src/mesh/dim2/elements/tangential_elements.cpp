#include "IO/fortranio/interface.hpp"
#include "enumerations/interface.hpp"
#include "mesh/dim2/elements/elements.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::elements::tangential_elements<specfem::dimension::type::dim2>::
    tangential_elements(const int nnodes_tangential_curve) {
  if (nnodes_tangential_curve > 0) {
    this->x = specfem::kokkos::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::x", nnodes_tangential_curve);
    this->y = specfem::kokkos::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::y", nnodes_tangential_curve);
  } else {
    this->x = specfem::kokkos::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::x", 1);
    this->y = specfem::kokkos::HostView1d<type_real>(
        "specfem::mesh::tangential_nodes::y", 1);
  }

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      this->x(inum) = 0.0;
      this->y(inum) = 0.0;
    }
  } else {
    this->x(0) = 0.0;
    this->y(0) = 0.0;
  }
  return;
}
