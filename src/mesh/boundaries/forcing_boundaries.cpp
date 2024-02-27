#include "IO/fortranio/interface.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "specfem_mpi/interface.hpp"
#include "utilities.cpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::boundaries::forcing_boundary::forcing_boundary(
    const int nelement_acforcing) {
  if (nelement_acforcing > 0) {
    this->numacforcing = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    this->codeacforcing = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing, 4);
    this->typeacforcing = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    this->ibegin_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge1", nelement_acforcing);
    this->ibegin_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge2", nelement_acforcing);
    this->ibegin_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge3", nelement_acforcing);
    this->ibegin_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge4", nelement_acforcing);
    this->iend_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge1", nelement_acforcing);
    this->iend_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge2", nelement_acforcing);
    this->iend_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge3", nelement_acforcing);
    this->iend_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge4", nelement_acforcing);
    this->ib_bottom = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_bottom", nelement_acforcing);
    this->ib_top = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_top", nelement_acforcing);
    this->ib_right = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_right", nelement_acforcing);
    this->ib_left = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_left", nelement_acforcing);
  } else {
    this->numacforcing = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    this->codeacforcing = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", 1, 1);
    this->typeacforcing = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    this->ibegin_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge1", 1);
    this->ibegin_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge2", 1);
    this->ibegin_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge3", 1);
    this->ibegin_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ibegin_edge4", 1);
    this->iend_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge1", 1);
    this->iend_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge2", 1);
    this->iend_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge3", 1);
    this->iend_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::iend_edge4", 1);
    this->ib_bottom = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_bottom", 1);
    this->ib_top = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_top", 1);
    this->ib_right = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_right", 1);
    this->ib_left = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::forcing_boundary::ib_left", 1);
  }

  if (nelement_acforcing > 0) {
    for (int n = 0; n < nelement_acforcing; n++) {
      this->numacforcing(n) = 0;
      this->typeacforcing(n) = 0;
      this->ibegin_edge1(n) = 0;
      this->ibegin_edge2(n) = 0;
      this->ibegin_edge3(n) = 0;
      this->ibegin_edge4(n) = 0;
      this->iend_edge1(n) = 0;
      this->iend_edge2(n) = 0;
      this->iend_edge3(n) = 0;
      this->iend_edge4(n) = 0;
      this->ib_bottom(n) = 0;
      this->ib_left(n) = 0;
      this->ib_top(n) = 0;
      this->ib_right(n) = 0;
      for (int i = 0; i < 4; i++) {
        this->codeacforcing(n, i) = false;
      }
    }
  } else {
    this->numacforcing(1) = 0;
    this->typeacforcing(1) = 0;
    this->ibegin_edge1(1) = 0;
    this->ibegin_edge2(1) = 0;
    this->ibegin_edge3(1) = 0;
    this->ibegin_edge4(1) = 0;
    this->iend_edge1(1) = 0;
    this->iend_edge2(1) = 0;
    this->iend_edge3(1) = 0;
    this->iend_edge4(1) = 0;
    this->ib_bottom(1) = 0;
    this->ib_left(1) = 0;
    this->ib_top(1) = 0;
    this->ib_right(1) = 0;
    this->codeacforcing(1, 1) = false;
  }
  return;
}

specfem::mesh::boundaries::forcing_boundary::forcing_boundary(
    std::ifstream &stream, const int nelement_acforcing, const int nspec,
    const specfem::MPI::MPI *mpi) {
  bool codeacread1 = true, codeacread2 = true, codeacread3 = true,
       codeacread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numacread, typeacread;

  *this = specfem::mesh::boundaries::forcing_boundary(nelement_acforcing);

  if (nelement_acforcing > 0) {
    for (int inum = 0; inum < nelement_acforcing; inum++) {
      specfem::IO::fortran_read_line(stream, &numacread, &codeacread1,
                                     &codeacread2, &codeacread3, &codeacread4,
                                     &typeacread, &iedgeread);
      std::vector<bool> codeacread(4, false);
      if (numacread < 1 || numacread > nspec) {
        std::runtime_error("Wrong absorbing element number");
      }
      this->numacforcing(inum) = numacread - 1;
      this->typeacforcing(inum) = typeacread;
      codeacread[0] = codeacread1;
      codeacread[1] = codeacread2;
      codeacread[2] = codeacread3;
      codeacread[3] = codeacread4;
      if (std::count(codeacread.begin(), codeacread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one acoustic forcing "
                                 "per acoustic forcing line cited");
      }
      this->codeacforcing(inum, 0) = codeacread[0];
      this->codeacforcing(inum, 1) = codeacread[1];
      this->codeacforcing(inum, 2) = codeacread[2];
      this->codeacforcing(inum, 3) = codeacread[3];
      this->ibegin_edge1(inum) = iedgeread[0];
      this->iend_edge1(inum) = iedgeread[1];
      this->ibegin_edge2(inum) = iedgeread[2];
      this->iend_edge2(inum) = iedgeread[3];
      this->ibegin_edge3(inum) = iedgeread[4];
      this->iend_edge3(inum) = iedgeread[5];
      this->ibegin_edge4(inum) = iedgeread[6];
      this->iend_edge4(inum) = iedgeread[7];
    }
  }

  // populate ib_bottom, ib_top, ib_left, ib_right arrays
  calculate_ib(this->codeacforcing, this->ib_bottom, this->ib_top,
               this->ib_left, this->ib_right, nelement_acforcing);

  return;
}
