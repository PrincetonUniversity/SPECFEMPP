#include "fortranio/interface.hpp"
#include "mesh/surfaces/surfaces.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::surfaces::acoustic_free_surface::acoustic_free_surface(
    const int nelem_acoustic_surface) {
  if (nelem_acoustic_surface > 0) {
    this->numacfree_surface = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::numacfree_surface",
        nelem_acoustic_surface);
    this->typeacfree_surface = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::typeacfree_surface",
        nelem_acoustic_surface);
    this->e1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e1", nelem_acoustic_surface);
    this->e2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::e2", nelem_acoustic_surface);
    this->ixmin = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmin", nelem_acoustic_surface);
    this->ixmax = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ixmax", nelem_acoustic_surface);
    this->izmin = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmin", nelem_acoustic_surface);
    this->izmax = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::izmax", nelem_acoustic_surface);
  }
  return;
}

specfem::mesh::surfaces::acoustic_free_surface::acoustic_free_surface(
    std::ifstream &stream, const int nelem_acoustic_surface,
    const specfem::MPI::MPI *mpi) {

  std::vector<int> acfree_edge(4, 0);
  *this =
      specfem::mesh::surfaces::acoustic_free_surface(nelem_acoustic_surface);

  if (nelem_acoustic_surface > 0) {
    for (int inum = 0; inum < nelem_acoustic_surface; inum++) {
      specfem::fortran_IO::fortran_read_line(stream, &acfree_edge);
      this->numacfree_surface(inum) = acfree_edge[0];
      this->typeacfree_surface(inum) = acfree_edge[1];
      this->e1(inum) = acfree_edge[2];
      this->e2(inum) = acfree_edge[3];
    }
  }

  mpi->sync_all();
  return;
}
