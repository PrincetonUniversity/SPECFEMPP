#include "io/fortranio/interface.hpp"
#include "mesh/dim2/boundaries/boundaries.hpp"

#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::forcing_boundary<specfem::dimension::type::dim2>::
    forcing_boundary(const int nelement_acforcing) {

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
    this->numacforcing(0) = 0;
    this->typeacforcing(0) = 0;
    this->ibegin_edge1(0) = 0;
    this->ibegin_edge2(0) = 0;
    this->ibegin_edge3(0) = 0;
    this->ibegin_edge4(0) = 0;
    this->iend_edge1(0) = 0;
    this->iend_edge2(0) = 0;
    this->iend_edge3(0) = 0;
    this->iend_edge4(0) = 0;
    this->ib_bottom(0) = 0;
    this->ib_left(0) = 0;
    this->ib_top(0) = 0;
    this->ib_right(0) = 0;
    this->codeacforcing(0, 0) = false;
  }
  return;
}
