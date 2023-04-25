#include "fortran_IO.h"
#include "mesh/elements/elements.hpp"
#include "specfem_mpi.h"

specfem::mesh::elements::tangential_elements::tangential_elements(
    const int nnodes_tangential_curve) {
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
    this->x(1) = 0.0;
    this->y(1) = 0.0;
  }
  return;
}

specfem::mesh::elements::tangential_elements::tangential_elements(
    std::ifstream &stream, const int nnodes_tangential_curve) {
  type_real xread, yread;

  *this = specfem::mesh::elements::tangential_elements(nnodes_tangential_curve);

  specfem::fortran_IO::fortran_read_line(stream, &this->force_normal_to_surface,
                                         &this->rec_normal_to_surface);

  if (nnodes_tangential_curve > 0) {
    for (int inum = 0; inum < nnodes_tangential_curve; inum++) {
      specfem::fortran_IO::fortran_read_line(stream, &xread, &yread);
      this->x(inum) = xread;
      this->y(inum) = yread;
    }
  } else {
    this->force_normal_to_surface = false;
    this->rec_normal_to_surface = false;
  }

  return;
}
