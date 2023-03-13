#include "../include/boundaries.h"
#include "../include/fortran_IO.h"
#include "../include/specfem_mpi.h"
#include <Kokkos_Core.hpp>
#include <vector>

// Find corner elements of the absorbing boundary
void find_corners(const specfem::kokkos::HostView1d<int> numabs,
                  const specfem::kokkos::HostView2d<bool> codeabs,
                  specfem::kokkos::HostView2d<bool> codeabscorner,
                  const int num_abs_boundary_faces,
                  const specfem::MPI::MPI *mpi) {
  int ncorner = 0;
  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (codeabs(inum, 0)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 1) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 2) = true;
              ncorner++;
            }
          }
        }
      }
    }
    if (codeabs(inum, 2)) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (numabs(inum) == numabs(inum_duplicate)) {
            if (codeabs(inum_duplicate, 3)) {
              codeabscorner(inum, 3) = true;
              ncorner++;
            }
            if (codeabs(inum_duplicate, 1)) {
              codeabscorner(inum, 4) = true;
              ncorner++;
            }
          }
        }
      }
    }
  }

  int ncorner_all = mpi->reduce(ncorner, specfem::MPI::sum);
  if (mpi->get_rank() == 0)
    assert(ncorner_all <= 4);
}

void calculate_ib(const specfem::kokkos::HostView2d<bool> code,
                  specfem::kokkos::HostView1d<int> ib_bottom,
                  specfem::kokkos::HostView1d<int> ib_top,
                  specfem::kokkos::HostView1d<int> ib_left,
                  specfem::kokkos::HostView1d<int> ib_right,
                  const int nelements) {

  int nspec_left = 0, nspec_right = 0, nspec_top = 0, nspec_bottom = 0;
  for (int inum = 0; inum < nelements; inum++) {
    if (code(inum, 0)) {
      ib_bottom(inum) = nspec_bottom;
      nspec_bottom++;
    } else if (code(inum, 1)) {
      ib_right(inum) = nspec_right;
      nspec_right++;
    } else if (code(inum, 2)) {
      ib_top(inum) = nspec_top;
      nspec_top++;
    } else if (code(inum, 3)) {
      ib_left(inum) = nspec_left;
      nspec_left++;
    } else {
      throw std::runtime_error("Incorrect acoustic boundary element type read");
    }
  }

  assert(nspec_left + nspec_right + nspec_bottom + nspec_top == nelements);
}

specfem::boundaries::absorbing_boundary::absorbing_boundary(
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

specfem::boundaries::forcing_boundary::forcing_boundary(
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

specfem::boundaries::absorbing_boundary::absorbing_boundary(
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

  *this = specfem::boundaries::absorbing_boundary(num_abs_boundary_faces);

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

specfem::boundaries::forcing_boundary::forcing_boundary(
    std::ifstream &stream, const int nelement_acforcing, const int nspec,
    const specfem::MPI::MPI *mpi) {
  bool codeacread1 = true, codeacread2 = true, codeacread3 = true,
       codeacread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numacread, typeacread;

  *this = specfem::boundaries::forcing_boundary(nelement_acforcing);

  if (nelement_acforcing > 0) {
    for (int inum = 0; inum < nelement_acforcing; inum++) {
      specfem::fortran_IO::fortran_read_line(
          stream, &numacread, &codeacread1, &codeacread2, &codeacread3,
          &codeacread4, &typeacread, &iedgeread);
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
