#include "specfem/algorithms/locate_point.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.tpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

specfem::point::local_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh) {

  // Extract mesh data and delegate to core implementation
  return specfem::algorithms::locate_point_impl::locate_point_core(
      mesh.graph(), coordinates, mesh.h_coord, mesh.h_control_node_coord,
      mesh.ngnod);
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

std::pair<type_real, bool> specfem::algorithms::locate_point_on_edge(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &mesh_entity) {

  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                     mesh_entity)) {
    throw std::runtime_error(
        "locate_point_on_edge mesh_entity must be an edge. Found a corner.");
    return { 0, false };
  }
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", mesh.ngnod);
  for (int i = 0; i < mesh.ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coord(0, ispec, i);
    coorg(i).z = mesh.h_control_node_coord(1, ispec, i);
  }

  // initial guess of 0 (center of edge)
  return specfem::algorithms::locate_point_impl::get_local_edge_coordinate(
      coordinates, coorg, mesh_entity, 0);
}

specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::algorithms::locate_point_on_edge(
    const type_real &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &mesh_entity) {
  if (specfem::mesh_entity::contains(specfem::mesh_entity::dim2::corners,
                                     mesh_entity)) {
    throw std::runtime_error(
        "locate_point_on_edge mesh_entity must be an edge. Found a corner.");
    return { 0, 0 };
  }
  const auto [xi, gamma] = [&]() -> std::pair<type_real, type_real> {
    if (mesh_entity == specfem::mesh_entity::dim2::type::bottom) {
      return { coordinate, -1 };
    } else if (mesh_entity == specfem::mesh_entity::dim2::type::right) {
      return { 1, coordinate };
    } else if (mesh_entity == specfem::mesh_entity::dim2::type::top) {
      return { coordinate, 1 };
    } else {
      return { -1, coordinate };
    }
  }();

  // interpolating the entire element is not the most efficient way to do this.
  // consider a codimension 1 interpolation in the future.

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
