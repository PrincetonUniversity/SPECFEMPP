#include "specfem/algorithms/locate_point.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"

specfem::point::local_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  // Extract mesh data and delegate to core implementation
  return locate_point_impl::locate_point_core(
      coordinates, mesh.h_coord, mesh.h_index_mapping,
      mesh.h_control_node_coordinates, mesh.ngnod, mesh.element_grid.ngllx);
}

specfem::point::global_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coordinates(ispec, i, 0);
    coorg(i).y = mesh.h_control_node_coordinates(ispec, i, 1);
    coorg(i).z = mesh.h_control_node_coordinates(ispec, i, 2);
  }

  return jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
}

// Except for the tests this function is not used in the codebase.
specfem::point::global_coordinates<specfem::dimension::type::dim3>
specfem::algorithms::locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, ngnod), [&](const int i) {
        coorg(i).x = mesh.h_control_node_coordinates(ispec, i, 0);
        coorg(i).y = mesh.h_control_node_coordinates(ispec, i, 1);
        coorg(i).z = mesh.h_control_node_coordinates(ispec, i, 2);
      });

  team_member.team_barrier();

  return jacobian::compute_locations(coorg, ngnod, xi, eta, gamma);
}
