#include "specfem/algorithms/locate_point.hpp"
#include "specfem/algorithms/locate_point/locate_point_impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/mesh_entity/dim3/mesh_entity.hpp"
#include "specfem/point.hpp"

specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
specfem::algorithms::locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  // Extract mesh data and delegate to core implementation
  return locate_point_impl::locate_point_core(
      coordinates, mesh.h_coord, mesh.h_index_mapping,
      mesh.h_control_node_coordinates, mesh.ngnod, mesh.element_grid.ngllx);
}

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::algorithms::locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::element::dimension_tag::dim3> *,
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
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
specfem::algorithms::locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &mesh) {

  const int ispec = coordinate.ispec;
  const type_real xi = coordinate.xi;
  const type_real eta = coordinate.eta;
  const type_real gamma = coordinate.gamma;

  const int ngnod = mesh.ngnod;

  const Kokkos::View<
      point::global_coordinates<specfem::element::dimension_tag::dim3> *,
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

std::pair<specfem::algorithms::facial_coordinate_type<
              specfem::element::dimension_tag::dim3>,
          bool>
locate_point_on_face(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint) {

  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      constraint)) {
    throw std::runtime_error("locate_point_on_face constraint must be a face "
                             "(edges are currently unsupported).");
    return { { 0, 0 }, false };
  }
  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", mesh.ngnod);
  for (int i = 0; i < mesh.ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coordinates(0, ispec, i);
    coorg(i).y = mesh.h_control_node_coordinates(1, ispec, i);
    coorg(i).z = mesh.h_control_node_coordinates(2, ispec, i);
  }

  // initial guess of 0 (center of edge)
  return specfem::algorithms::locate_point_impl::get_local_face_coordinate(
      coordinates, coorg, constraint, { 0, 0 });
}

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point_on_face(
    const specfem::algorithms::facial_coordinate_type<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint) {
  if (!specfem::mesh_entity::contains(specfem::mesh_entity::dim3::faces,
                                      constraint)) {
    throw std::runtime_error("locate_point_on_face constraint must be a face "
                             "(edges are currently unsupported).");
    return { 0, 0, 0 };
  }
  const auto [xi, gamma,
              eta] = [&]() -> std::tuple<type_real, type_real, type_real> {
    if (constraint == specfem::mesh_entity::dim3::type::bottom) {
      return { coordinates.first, coordinates.second, -1 };
    } else if (constraint == specfem::mesh_entity::dim3::type::right) {
      return { 1, coordinates.first, coordinates.second };
    } else if (constraint == specfem::mesh_entity::dim3::type::top) {
      return { coordinates.first, coordinates.second, 1 };
    } else if (constraint == specfem::mesh_entity::dim3::type::left) {
      return { -1, coordinates.first, coordinates.second };
    } else if (constraint == specfem::mesh_entity::dim3::type::front) {
      return { coordinates.first, -1, coordinates.second };
    } else { // back
      return { coordinates.first, 1, coordinates.second };
    }
  }();

  // interpolating the entire element is not the most efficient way to do this.
  // consider a codimension 1 interpolation in the future.

  const int ngnod = mesh.ngnod;

  const Kokkos::View<specfem::point::global_coordinates<
                         specfem::element::dimension_tag::dim3> *,
                     Kokkos::HostSpace>
      coorg("coorg", ngnod);

  for (int i = 0; i < ngnod; i++) {
    coorg(i).x = mesh.h_control_node_coordinates(0, ispec, i);
    coorg(i).y = mesh.h_control_node_coordinates(1, ispec, i);
    coorg(i).z = mesh.h_control_node_coordinates(2, ispec, i);
  }

  return specfem::jacobian::compute_locations(coorg, ngnod, xi, gamma, eta);
}
