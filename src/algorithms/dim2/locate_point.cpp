#include "algorithms/locate_point.hpp"
#include "Serial/Kokkos_Serial_Parallel_Range.hpp"
#include "algorithms/locate_point_impl.hpp"
#include "algorithms/locate_point_impl.tpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

specfem::point::local_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh) {

  // Extract mesh data and delegate to core implementation
  if (mesh.adjacency_graph_empty()) {
    return specfem::algorithms::locate_point_impl::locate_point_core(
        coordinates, mesh.h_coord, mesh.h_index_mapping,
        mesh.h_control_node_coord, mesh.ngnod, mesh.ngllx);
  } else {
    return specfem::algorithms::locate_point_impl::locate_point_core(
        mesh.graph(), coordinates, mesh.h_coord, mesh.h_control_node_coord,
        mesh.ngnod);
  }
}

specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coord(0, ispec, i);
    coorg(i).z = mesh.h_control_node_coord(1, ispec, i);
  }

  return jacobian::compute_locations(coorg, ngnod, xi, gamma);
}

// Except for the tests this function is not used in the codebase.
specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ngnod),
                       [&](const int i) {
                         coorg(i).x = mesh.h_control_node_coord(0, ispec, i);
                         coorg(i).z = mesh.h_control_node_coord(1, ispec, i);
                       });

  team_member.team_barrier();

  return jacobian::compute_locations(coorg, ngnod, xi, gamma);
}

type_real specfem::algorithms::locate_point_on_edge(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::type &mesh_entity) {

  if (mesh_entity == specfem::mesh_entity::type::bottom_left ||
      mesh_entity == specfem::mesh_entity::type::bottom_right ||
      mesh_entity == specfem::mesh_entity::type::top_left ||
      mesh_entity == specfem::mesh_entity::type::top_right) {
    throw std::runtime_error(
        "locate_point_on_edge mesh_entity must be an edge. Found a corner.");
    return 0;
  }
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", mesh.ngnod);
  for (int i = 0; i < mesh.ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coord(0, ispec, i);
    coorg(i).z = mesh.h_control_node_coord(1, ispec, i);
  }

  type_real coord_guess = 0;
  coord_guess =
      specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
          coordinates, coorg, mesh_entity, coord_guess);

  return coord_guess;
}
