#include "../include/boundaries.h"
#include <Kokkos_Core.hpp>

specfem::boundaries::absorbing_boundary::absorbing_boundary(
    const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    this->numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", num_abs_boundary_faces);
    this->abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type",
        num_abs_boundary_faces);
    this->ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1",
        num_abs_boundary_faces);
    this->ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2",
        num_abs_boundary_faces);
    this->ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3",
        num_abs_boundary_faces);
    this->ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4",
        num_abs_boundary_faces);
    this->iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1",
        num_abs_boundary_faces);
    this->iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2",
        num_abs_boundary_faces);
    this->iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3",
        num_abs_boundary_faces);
    this->iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4",
        num_abs_boundary_faces);
    this->ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", num_abs_boundary_faces);
    this->ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", num_abs_boundary_faces);
    this->ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", num_abs_boundary_faces);
    this->ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", num_abs_boundary_faces);
  } else {
    this->numabs = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::numabs", 1);
    this->abs_boundary_type = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::abs_boundary_type", 1);
    this->ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    this->ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    this->ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    this->ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    this->iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    this->iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    this->iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    this->iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    this->ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    this->ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    this->ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    this->ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
  }

  if (num_abs_boundary_faces > 0) {
    this->codeabs =
        specfem::HostView2d<bool>("specfem::mesh::absorbing_boundary::codeabs",
                                  num_abs_boundary_faces, 4);
    this->codeabscorner = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs_corner",
        num_abs_boundary_faces, 4);
  } else {
    this->codeabs = specfem::HostView2d<bool>(
        "specfem::mesh::absorbing_boundary::codeabs", 1, 1);
    this->codeabscorner = specfem::HostView2d<bool>(
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
    this->numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    this->codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing, 4);
    this->typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", nelement_acforcing);
    this->ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", nelement_acforcing);
    this->ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", nelement_acforcing);
    this->ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", nelement_acforcing);
    this->ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", nelement_acforcing);
    this->iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", nelement_acforcing);
    this->iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", nelement_acforcing);
    this->iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", nelement_acforcing);
    this->iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", nelement_acforcing);
    this->ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", nelement_acforcing);
    this->ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", nelement_acforcing);
    this->ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", nelement_acforcing);
    this->ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", nelement_acforcing);
  } else {
    this->numacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    this->codeacforcing = specfem::HostView2d<bool>(
        "specfem::mesh::forcing_boundary::numacforcing", 1, 1);
    this->typeacforcing = specfem::HostView1d<int>(
        "specfem::mesh::forcing_boundary::numacforcing", 1);
    this->ibegin_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge1", 1);
    this->ibegin_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge2", 1);
    this->ibegin_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge3", 1);
    this->ibegin_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ibegin_edge4", 1);
    this->iend_edge1 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge1", 1);
    this->iend_edge2 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge2", 1);
    this->iend_edge3 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge3", 1);
    this->iend_edge4 = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::iend_edge4", 1);
    this->ib_bottom = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_bottom", 1);
    this->ib_top = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_top", 1);
    this->ib_right = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_right", 1);
    this->ib_left = specfem::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ib_left", 1);
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
