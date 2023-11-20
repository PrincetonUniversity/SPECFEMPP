#include "mesh/boundaries/acoustic_free_surface.hpp"
#include "fortranio/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>

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
 * @return specfem::enums::boundaries::type type of the boundary
 */
specfem::enums::boundaries::type
get_boundary_type(const int type, const int e1, const int e2,
                  const view_type &control_nodes) {

  // if this is a node type
  if (type == 1) {
    if (e1 == control_nodes(0)) {
      return specfem::enums::boundaries::type::BOTTOM_LEFT;
    } else if (e1 == control_nodes(1)) {
      return specfem::enums::boundaries::type::BOTTOM_RIGHT;
    } else if (e1 == control_nodes(2)) {
      return specfem::enums::boundaries::type::TOP_RIGHT;
    } else if (e1 == control_nodes(3)) {
      return specfem::enums::boundaries::type::TOP_LEFT;
    } else {
      throw std::invalid_argument(
          "Error: Could not generate type of acoustic free surface boundary");
    }
  } else {
    if ((e1 == control_nodes(0) && e2 == control_nodes(1)) ||
        (e1 == control_nodes(1) && e2 == control_nodes(0))) {
      return specfem::enums::boundaries::type::BOTTOM;
    } else if ((e1 == control_nodes(0) && e2 == control_nodes(3)) ||
               (e1 == control_nodes(3) && e2 == control_nodes(0))) {
      return specfem::enums::boundaries::type::LEFT;
    } else if ((e1 == control_nodes(1) && e2 == control_nodes(2)) ||
               (e1 == control_nodes(2) && e2 == control_nodes(1))) {
      return specfem::enums::boundaries::type::RIGHT;
    } else if ((e1 == control_nodes(2) && e2 == control_nodes(3)) ||
               (e1 == control_nodes(3) && e2 == control_nodes(2))) {
      return specfem::enums::boundaries::type::TOP;
    } else {
      throw std::invalid_argument(
          "Error: Could not generate type of acoustic free surface boundary");
    }
  }
}

specfem::mesh::boundaries::acoustic_free_surface::acoustic_free_surface(
    const int nelem_acoustic_surface)
    : nelem_acoustic_surface(nelem_acoustic_surface) {
  if (nelem_acoustic_surface > 0) {
    this->ispec_acoustic_surface = specfem::kokkos::HostView1d<int>(
        "specfem::mesh::acoustic_free_surface::ispec_acoustic_surface",
        nelem_acoustic_surface);
    this->type = specfem::kokkos::HostView1d<specfem::enums::boundaries::type>(
        "specfem::mesh::acoustic_free_surface::type", nelem_acoustic_surface);
  }
  return;
}

specfem::mesh::boundaries::acoustic_free_surface::acoustic_free_surface(
    std::ifstream &stream, const int &nelem_acoustic_surface,
    const specfem::kokkos::HostView2d<int> &knods,
    const specfem::MPI::MPI *mpi) {

  std::vector<int> acfree_edge(4, 0);
  *this =
      specfem::mesh::boundaries::acoustic_free_surface(nelem_acoustic_surface);

  if (nelem_acoustic_surface > 0) {
    for (int inum = 0; inum < nelem_acoustic_surface; inum++) {
      specfem::fortran_IO::fortran_read_line(stream, &acfree_edge);
      this->ispec_acoustic_surface(inum) = acfree_edge[0] - 1;
      const auto control_nodes = Kokkos::subview(
          knods, Kokkos::ALL, this->ispec_acoustic_surface(inum));
      this->type(inum) = get_boundary_type(acfree_edge[1], acfree_edge[2] - 1,
                                           acfree_edge[3] - 1, control_nodes);
    }
  }

  mpi->sync_all();
  return;
}
