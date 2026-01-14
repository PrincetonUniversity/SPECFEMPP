#include "io/mesh/impl/fortran/dim2/read_boundaries.hpp"
#include "io/fortranio/interface.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh.hpp"

#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

static std::tuple<
    specfem::kokkos::HostView1d<int>,
    specfem::kokkos::HostView1d<specfem::mesh_entity::dim2::type> >
find_corners(const specfem::kokkos::HostView1d<int> ispec_edge,
             const specfem::kokkos::HostView1d<specfem::mesh_entity::dim2::type>
                 type_edge) {

  int ncorner = 0;
  int num_abs_boundary_faces = ispec_edge.extent(0);
  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (type_edge(inum) == specfem::mesh_entity::dim2::type::bottom) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
            if (type_edge(inum) == specfem::mesh_entity::dim2::type::left) {
              ncorner++;
            }
            if (type_edge(inum) == specfem::mesh_entity::dim2::type::right) {
              ncorner++;
            }
          }
        }
      }
      if (type_edge(inum) == specfem::mesh_entity::dim2::type::top) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
              if (type_edge(inum) == specfem::mesh_entity::dim2::type::left) {
                ncorner++;
              }
              if (type_edge(inum) == specfem::mesh_entity::dim2::type::right) {
                ncorner++;
              }
            }
          }
        }
      }
    }
  }

  specfem::kokkos::HostView1d<int> ispec_corners(
      "specfem:io::mesh::impl::fortran::read_boundaries::find_corners::ispec_"
      "corners",
      ncorner);

  specfem::kokkos::HostView1d<specfem::mesh_entity::dim2::type> type_corners(
      "specfem:io::mesh::impl::fortran::read_boundaries::find_corners::type_"
      "corners",
      ncorner);

  int icorner = 0;

  for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
    if (type_edge(inum) == specfem::mesh_entity::dim2::type::bottom) {
      for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
           inum_duplicate++) {
        if (inum != inum_duplicate) {
          if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
            if (type_edge(inum) == specfem::mesh_entity::dim2::type::left) {
              ispec_corners(icorner) = ispec_edge(inum);
              type_corners(icorner) =
                  specfem::mesh_entity::dim2::type::bottom_left;
              icorner++;
            }
            if (type_edge(inum) == specfem::mesh_entity::dim2::type::right) {
              ispec_corners(icorner) = ispec_edge(inum);
              type_corners(icorner) =
                  specfem::mesh_entity::dim2::type::bottom_right;
              icorner++;
            }
          }
        }
      }
      if (type_edge(inum) == specfem::mesh_entity::dim2::type::top) {
        for (int inum_duplicate = 0; inum_duplicate < num_abs_boundary_faces;
             inum_duplicate++) {
          if (inum != inum_duplicate) {
            if (ispec_edge(inum) == ispec_edge(inum_duplicate)) {
              if (type_edge(inum) == specfem::mesh_entity::dim2::type::left) {
                ispec_corners(icorner) = ispec_edge(inum);
                type_corners(icorner) =
                    specfem::mesh_entity::dim2::type::top_left;
                icorner++;
              }
              if (type_edge(inum) == specfem::mesh_entity::dim2::type::right) {
                ispec_corners(icorner) = ispec_edge(inum);
                type_corners(icorner) =
                    specfem::mesh_entity::dim2::type::top_right;
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

inline void calculate_ib(const specfem::kokkos::HostView2d<bool> code,
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

specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2>
read_absorbing_boundaries(std::ifstream &stream, int num_abs_boundary_faces,
                          const int nspec) {

  // Create base instance of the absorbing boundary
  specfem::mesh::absorbing_boundary<specfem::dimension::type::dim2>
      absorbing_boundary(num_abs_boundary_faces);

  // I have to do this because std::vector<bool> is a fake container type that
  // causes issues when getting a reference
  bool codeabsread1 = true, codeabsread2 = true, codeabsread3 = true,
       codeabsread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numabsread, typeabsread;
  if (num_abs_boundary_faces < 0) {
    specfem::Logger::warning("Read in negative nelemabs resetting to 0!");
    num_abs_boundary_faces = 0;
  }

  specfem::kokkos::HostView1d<specfem::mesh_entity::dim2::type> type_edge(
      "specfem::mesh::absorbing_boundary::type_edge", num_abs_boundary_faces);

  specfem::kokkos::HostView1d<int> ispec_edge(
      "specfem::mesh::absorbing_boundary::ispec_edge", num_abs_boundary_faces);

  if (num_abs_boundary_faces > 0) {
    for (int inum = 0; inum < num_abs_boundary_faces; inum++) {
      specfem::io::fortran_read_line(stream, &numabsread, &codeabsread1,
                                     &codeabsread2, &codeabsread3,
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
        type_edge(inum) = specfem::mesh_entity::dim2::type::bottom;

      if (codeabsread2)
        type_edge(inum) = specfem::mesh_entity::dim2::type::right;

      if (codeabsread3)
        type_edge(inum) = specfem::mesh_entity::dim2::type::top;

      if (codeabsread4)
        type_edge(inum) = specfem::mesh_entity::dim2::type::left;
    }

    // Find corner elements
    auto [ispec_corners, type_corners] = find_corners(ispec_edge, type_edge);

    const int nelements = ispec_corners.extent(0) + ispec_edge.extent(0);

    absorbing_boundary.nelements = nelements;

    absorbing_boundary.index_mapping = Kokkos::View<int *, Kokkos::HostSpace>(
        "specfem::mesh::absorbing_boundary::index_mapping", nelements);

    absorbing_boundary.type =
        Kokkos::View<specfem::mesh_entity::dim2::type *, Kokkos::HostSpace>(
            "specfem::mesh::absorbing_boundary::type", nelements);
    // Populate ispec and type arrays

    for (int inum = 0; inum < ispec_edge.extent(0); inum++) {
      absorbing_boundary.index_mapping(inum) = ispec_edge(inum);
      absorbing_boundary.type(inum) = type_edge(inum);
    }

    for (int inum = 0; inum < ispec_corners.extent(0); inum++) {
      absorbing_boundary.index_mapping(inum + ispec_edge.extent(0)) =
          ispec_corners(inum);
      absorbing_boundary.type(inum + ispec_edge.extent(0)) = type_corners(inum);
    }
  } else {
    absorbing_boundary.nelements = 0;
  }

  return absorbing_boundary;
}

using view_type =
    Kokkos::Subview<specfem::kokkos::HostView2d<int>,
                    std::remove_const_t<decltype(Kokkos::ALL)>, int>;

/**
 * @brief Get the type of boundary
 *
 * @param type int indicating if the boundary is edge of node
 * @param e1 control node index for the starting node of the if the boundary is
 * edge else control node index of the node if the boundary is node
 * @param e2 control node index for the ending node of the if the boundary is
 * edge
 * @return specfem::mesh_entity::dim2::type type of the boundary
 */
specfem::mesh_entity::dim2::type
get_boundary_type(const int type, const int e1, const int e2,
                  const view_type &control_nodes) {
  // if this is a node type
  if (type == 1) {
    if (e1 == control_nodes(0)) {
      return specfem::mesh_entity::dim2::type::bottom_left;
    } else if (e1 == control_nodes(1)) {
      return specfem::mesh_entity::dim2::type::bottom_right;
    } else if (e1 == control_nodes(2)) {
      return specfem::mesh_entity::dim2::type::top_right;
    } else if (e1 == control_nodes(3)) {
      return specfem::mesh_entity::dim2::type::top_left;
    } else {
      throw std::invalid_argument(
          "Error: Could not generate type of acoustic free surface boundary");
    }
  } else {
    if ((e1 == control_nodes(0) && e2 == control_nodes(1)) ||
        (e1 == control_nodes(1) && e2 == control_nodes(0))) {
      return specfem::mesh_entity::dim2::type::bottom;
    } else if ((e1 == control_nodes(0) && e2 == control_nodes(3)) ||
               (e1 == control_nodes(3) && e2 == control_nodes(0))) {
      return specfem::mesh_entity::dim2::type::left;
    } else if ((e1 == control_nodes(1) && e2 == control_nodes(2)) ||
               (e1 == control_nodes(2) && e2 == control_nodes(1))) {
      return specfem::mesh_entity::dim2::type::right;
    } else if ((e1 == control_nodes(2) && e2 == control_nodes(3)) ||
               (e1 == control_nodes(3) && e2 == control_nodes(2))) {
      return specfem::mesh_entity::dim2::type::top;
    } else {
      throw std::invalid_argument(
          "Error: Could not generate type of acoustic free surface boundary");
    }
  }
}

specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2>
read_acoustic_free_surface(
    std::ifstream &stream, const int &nelem_acoustic_surface,
    const Kokkos::View<int **, Kokkos::HostSpace> knods) {

  std::vector<int> acfree_edge(4, 0);
  specfem::mesh::acoustic_free_surface<specfem::dimension::type::dim2>
      acoustic_free_surface(nelem_acoustic_surface);

  if (nelem_acoustic_surface > 0) {
    for (int inum = 0; inum < nelem_acoustic_surface; inum++) {
      specfem::io::fortran_read_line(stream, &acfree_edge);
      acoustic_free_surface.index_mapping(inum) = acfree_edge[0] - 1;
      const auto control_nodes = Kokkos::subview(
          knods, Kokkos::ALL, acoustic_free_surface.index_mapping(inum));
      acoustic_free_surface.type(inum) =
          get_boundary_type(acfree_edge[1], acfree_edge[2] - 1,
                            acfree_edge[3] - 1, control_nodes);
    }
  }

  specfem::MPI::sync_all();

  return acoustic_free_surface;
}

specfem::mesh::forcing_boundary<specfem::dimension::type::dim2>
read_forcing_boundaries(std::ifstream &stream, const int nelement_acforcing,
                        const int nspec) {

  bool codeacread1 = true, codeacread2 = true, codeacread3 = true,
       codeacread4 = true;
  std::vector<int> iedgeread(8, 0);
  int numacread, typeacread;

  specfem::mesh::forcing_boundary<specfem::dimension::type::dim2>
      forcing_boundary(nelement_acforcing);

  if (nelement_acforcing > 0) {
    for (int inum = 0; inum < nelement_acforcing; inum++) {
      specfem::io::fortran_read_line(stream, &numacread, &codeacread1,
                                     &codeacread2, &codeacread3, &codeacread4,
                                     &typeacread, &iedgeread);
      std::vector<bool> codeacread(4, false);
      if (numacread < 1 || numacread > nspec) {
        std::runtime_error("Wrong absorbing element number");
      }
      forcing_boundary.numacforcing(inum) = numacread - 1;
      forcing_boundary.typeacforcing(inum) = typeacread;
      codeacread[0] = codeacread1;
      codeacread[1] = codeacread2;
      codeacread[2] = codeacread3;
      codeacread[3] = codeacread4;
      if (std::count(codeacread.begin(), codeacread.end(), true) != 1) {
        throw std::runtime_error("must have one and only one acoustic forcing "
                                 "per acoustic forcing line cited");
      }
      forcing_boundary.codeacforcing(inum, 0) = codeacread[0];
      forcing_boundary.codeacforcing(inum, 1) = codeacread[1];
      forcing_boundary.codeacforcing(inum, 2) = codeacread[2];
      forcing_boundary.codeacforcing(inum, 3) = codeacread[3];
      forcing_boundary.ibegin_edge1(inum) = iedgeread[0];
      forcing_boundary.iend_edge1(inum) = iedgeread[1];
      forcing_boundary.ibegin_edge2(inum) = iedgeread[2];
      forcing_boundary.iend_edge2(inum) = iedgeread[3];
      forcing_boundary.ibegin_edge3(inum) = iedgeread[4];
      forcing_boundary.iend_edge3(inum) = iedgeread[5];
      forcing_boundary.ibegin_edge4(inum) = iedgeread[6];
      forcing_boundary.iend_edge4(inum) = iedgeread[7];
    }
  }

  // populate ib_bottom, ib_top, ib_left, ib_right arrays
  calculate_ib(forcing_boundary.codeacforcing, forcing_boundary.ib_bottom,
               forcing_boundary.ib_top, forcing_boundary.ib_left,
               forcing_boundary.ib_right, nelement_acforcing);

  return forcing_boundary;
}

specfem::mesh::boundaries<specfem::dimension::type::dim2>
specfem::io::mesh::impl::fortran::dim2::read_boundaries(
    std::ifstream &stream, const int nspec, const int n_absorbing,
    const int n_acoustic_surface, const int n_acforcing,
    const Kokkos::View<int **, Kokkos::HostSpace> knods) {

  // Read absorbing boundaries
  auto absorbing_boundary =
      read_absorbing_boundaries(stream, n_absorbing, nspec);
  // Read acoustic free surface
  auto acoustic_free_surface =
      read_acoustic_free_surface(stream, n_acoustic_surface, knods);

  // Read forcing boundaries
  auto forcing_boundary = read_forcing_boundaries(stream, n_acforcing, nspec);
  return { absorbing_boundary, acoustic_free_surface, forcing_boundary };
}
