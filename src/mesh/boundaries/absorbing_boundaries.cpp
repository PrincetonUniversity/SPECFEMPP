#include "fortranio/interface.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

specfem::mesh::boundaries::absorbing_boundary::absorbing_boundary(
    const int num_abs_boundary_faces) {
  if (num_abs_boundary_faces > 0) {
    this->nelements = num_abs_boundary_faces;
    this->ispec = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ispec", num_abs_boundary_faces);
    this->type = specfem::kokkos::HostView1d<specfem::enums::boundaries::type>(
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

  specfem::kokkos::HostView1d<specfem::enums::boundaries::type> type_edge(
      "specfem::mesh::absorbing_boundary::type_edge", num_abs_boundary_faces);

  specfem::kokkos::HostView1d<int> ispec_edge(
      "specfem::mesh::absorbing_boundary::ispec_edge", num_abs_boundary_faces);

  if (num_abs_boundary_faces > 0) {
    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      specfem::fortran_IO::fortran_read_line(
          stream, &numabsread, &codeabsread1, &codeabsread2, &codeabsread3,
          &codeabsread4, &typeabsread, &iedgeread);
      if (numabsread < 1 || numabsread > nspec)
        throw std::runtime_error("Wrong absorbing element number");
      ispec_edge(inum) = numabsread - 1;
      std::vector<bool> codeabsread = { codeabsread1, codeabsread2,
                                        codeabsread3, codeabsread4 };
      if (std::count(codeabsread.begin(), codeabsread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one absorbing edge "
                                 "per absorbing line cited");
      }
      if (codeabsread1)
        type_edge(inum) = specfem::enums::boundaries::type::BOTTOM;

      if (codeabsread2)
        type_edge(inum) = specfem::enums::boundaries::type::RIGHT;

      if (codeabsread3)
        type_edge(inum) = specfem::enums::boundaries::type::TOP;

      if (codeabsread4)
        type_edge(inum) = specfem::enums::boundaries::type::LEFT;
    }

    // Find corner elements
    auto [ispec_corners, type_corners] = find_corners(ispec_edge, type_edge);

    const int nelements = ispec_corners.extent(0) + ispec_edge.extent(0);

    this->nelements = nelements;

    this->ispec = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::absorbing_boundary::ispec", nelements);

    this->type = specfem::kokkos::HostView1d<specfem::enums::boundaries::type>(
        "specfem::mesh::absorbing_boundary::type", nelements);

    // Populate ispec and type arrays

    for (int inum = 0; inum < ispec_edge.extent(0); inum++) {
      this->ispec(inum) = ispec_edge(inum);
      this->type(inum) = type_edge(inum);
    }

    for (int inum = 0; inum < ispec_corners.extent(0); inum++) {
      this->ispec(inum + ispec_edge.extent(0)) = ispec_corners(inum);
      this->type(inum + ispec_edge.extent(0)) = type_corners(inum);
    }
  } else {
    this->nelements = 0;
  }

  return;
}
