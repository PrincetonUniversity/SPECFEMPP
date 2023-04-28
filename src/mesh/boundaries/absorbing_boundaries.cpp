#include "fortranio/interface.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "specfem_mpi/interface.hpp"
#include "utilities.cpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::boundaries::absorbing_boundary::absorbing_boundary(
    const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    this->numabs = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", num_abs_boundary_faces);
    this->abs_boundary_type = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type",
        num_abs_boundary_faces);
    this->ibegin_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1",
        num_abs_boundary_faces);
    this->ibegin_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2",
        num_abs_boundary_faces);
    this->ibegin_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3",
        num_abs_boundary_faces);
    this->ibegin_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4",
        num_abs_boundary_faces);
    this->iend_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1",
        num_abs_boundary_faces);
    this->iend_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2",
        num_abs_boundary_faces);
    this->iend_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3",
        num_abs_boundary_faces);
    this->iend_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4",
        num_abs_boundary_faces);
    this->ib_bottom = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", num_abs_boundary_faces);
    this->ib_top = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", num_abs_boundary_faces);
    this->ib_right = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", num_abs_boundary_faces);
    this->ib_left = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", num_abs_boundary_faces);
  } else {
    this->numabs = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", 1);
    this->abs_boundary_type = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type", 1);
    this->ibegin_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    this->ibegin_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    this->ibegin_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    this->ibegin_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    this->iend_edge1 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    this->iend_edge2 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    this->iend_edge3 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    this->iend_edge4 = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    this->ib_bottom = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    this->ib_top = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    this->ib_right = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    this->ib_left = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (num_abs_boundary_faces > 0) {
    this->codeabs = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs", num_abs_boundary_faces,
        4);
    this->codeabscorner = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner",
        num_abs_boundary_faces, 4);
  } else {
    this->codeabs = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs", 1, 1);
    this->codeabscorner = specfem::kokkos::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner", 1, 1);
  }

  if (num_abs_boundary_faces > 0) {
    for (int n = 0; n < num_abs_boundary_faces; n++) {
      this->numabs(n) = 0;
      this->abs_boundary_type(n) = 0;
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
        this->codeabs(n, i) = false;
        this->codeabscorner(n, i) = false;
      }
    }
  } else {
    this->numabs(1) = 0;
    this->abs_boundary_type(1) = 0;
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
    this->codeabs(1, 1) = false;
    this->codeabscorner(1, 1) = false;
  }
  return;
}

specfem::mesh::boundaries::absorbing_boundary::absorbing_boundary(
    std::ifstream &stream, int num_abs_boundary_faces, const int nspec,
    const specfem::MPI::MPI *mpi) {

  // I have to do this because std::vector<bool> is a fake container type that
  // causes issues when getting a reference
  bool codeabsread1 = true, codeabsread2 = true, codeabsread3 = true,
       codeabsread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numabsread, typeabsread;
  if (num_abs_boundary_faces < 0) {
    mpi->cout("Warning: read in negative nelemabs resetting to 0!");
    num_abs_boundary_faces = 0;
  }

  *this = specfem::mesh::boundaries::absorbing_boundary(num_abs_boundary_faces);

  if (num_abs_boundary_faces > 0) {
    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      specfem::fortran_IO::fortran_read_line(
          stream, &numabsread, &codeabsread1, &codeabsread2, &codeabsread3,
          &codeabsread4, &typeabsread, &iedgeread);
      std::vector<bool> codeabsread(4, false);
      if (numabsread < 1 || numabsread > nspec)
        throw std::runtime_error("Wrong absorbing element number");
      this->numabs(inum) = numabsread - 1;
      this->abs_boundary_type(inum) = typeabsread;
      codeabsread[0] = codeabsread1;
      codeabsread[1] = codeabsread2;
      codeabsread[2] = codeabsread3;
      codeabsread[3] = codeabsread4;
      if (std::count(codeabsread.begin(), codeabsread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one absorbing edge "
                                 "per absorbing line cited");
      }
      this->codeabs(inum, 0) = codeabsread[0];
      this->codeabs(inum, 1) = codeabsread[1];
      this->codeabs(inum, 2) = codeabsread[2];
      this->codeabs(inum, 3) = codeabsread[3];
      this->ibegin_edge1(inum) = iedgeread[0];
      this->iend_edge1(inum) = iedgeread[1];
      this->ibegin_edge2(inum) = iedgeread[2];
      this->iend_edge2(inum) = iedgeread[3];
      this->ibegin_edge3(inum) = iedgeread[4];
      this->iend_edge3(inum) = iedgeread[5];
      this->ibegin_edge4(inum) = iedgeread[6];
      this->iend_edge4(inum) = iedgeread[7];
    }

    // Find corner elements
    find_corners(this->numabs, this->codeabs, this->codeabscorner,
                 num_abs_boundary_faces, mpi);

    // populate ib_bottom, ib_top, ib_left, ib_right arrays
    calculate_ib(this->codeabs, this->ib_bottom, this->ib_top, this->ib_left,
                 this->ib_right, num_abs_boundary_faces);
  }

  return;
}
