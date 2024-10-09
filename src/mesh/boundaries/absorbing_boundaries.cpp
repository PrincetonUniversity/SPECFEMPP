#include "IO/fortranio/interface.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::absorbing_boundary::absorbing_boundary(
    const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    this->nelements = num_abs_boundary_faces;
    this->index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::absorbing_boundary::index_mapping",
        num_abs_boundary_faces);
    this->type =
        Kokkos::View<specfem::enums::boundaries::type *, Kokkos::HostSpace>(
            "specfem::mesh::absorbing_boundary::type", num_abs_boundary_faces);
  } else {
    this->nelements = 0;
  }

  return;
}

static std::tuple<
    specfem::kokkos::HostView1d<int>,
    specfem::kokkos::HostView1d<specfem::enums::boundaries::type> >
find_corners(const specfem::kokkos::HostView1d<int> ispec_edge,
             const specfem::kokkos::HostView1d<specfem::enums::boundaries::type>
                 type_edge) {

  int ncorner = 0;
  int num_abs_boundary_faces = ispec_edge.extent(0);
  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (type_edge(inum) == specfem::enums::boundaries::type::BOTTOM) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
            if (type_edge(inum) == specfem::enums::boundaries::type::LEFT) {
              ncorner++;
            }
            if (type_edge(inum) == specfem::enums::boundaries::type::RIGHT) {
              ncorner++;
            }
          }
        }
      }
      if (type_edge(inum) == specfem::enums::boundaries::type::TOP) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
              if (type_edge(inum) == specfem::enums::boundaries::type::LEFT) {
                ncorner++;
              }
              if (type_edge(inum) == specfem::enums::boundaries::type::RIGHT) {
                ncorner++;
              }
            }
          }
        }
      }
    }
  }

  specfem::kokkos::HostView1d<int> ispec_corners(
      "specfem::mesh::absorbing_boundary::ispec_corners", ncorner);

  specfem::kokkos::HostView1d<specfem::enums::boundaries::type> type_corners(
      "specfem::mesh::absorbing_boundary::type_corners", ncorner);

  int icorner = 0;

  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (type_edge(inum) == specfem::enums::boundaries::type::BOTTOM) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
            if (type_edge(inum) == specfem::enums::boundaries::type::LEFT) {
              ispec_corners(icorner) = ispec_edge(inum);
              type_corners(icorner) =
                  specfem::enums::boundaries::type::BOTTOM_LEFT;
              icorner++;
            }
            if (type_edge(inum) == specfem::enums::boundaries::type::RIGHT) {
              ispec_corners(icorner) = ispec_edge(inum);
              type_corners(icorner) =
                  specfem::enums::boundaries::type::BOTTOM_RIGHT;
              icorner++;
            }
          }
        }
      }
      if (type_edge(inum) == specfem::enums::boundaries::type::TOP) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
              if (type_edge(inum) == specfem::enums::boundaries::type::LEFT) {
                ispec_corners(icorner) = ispec_edge(inum);
                type_corners(icorner) =
                    specfem::enums::boundaries::type::TOP_LEFT;
                icorner++;
              }
              if (type_edge(inum) == specfem::enums::boundaries::type::RIGHT) {
                ispec_corners(icorner) = ispec_edge(inum);
                type_corners(icorner) =
                    specfem::enums::boundaries::type::TOP_RIGHT;
                icorner++;
              }
            }
          }
        }
      }
    }
  }

  return std::make_tuple(ispec_corners, type_corners);
}
