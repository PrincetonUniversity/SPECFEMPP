#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <vector>

// Implementation details exposed for testing
namespace specfem::algorithms::locate_point_impl {

std::tuple<int, int, int> rough_location(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &global,
    const Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        coord);

std::vector<int> get_best_candidates(
    const int ispec_guess,
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        index_mapping);

std::tuple<type_real, type_real> get_local_coordinates(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &global,
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim2> *,
                       Kokkos::HostSpace> &coorg,
    type_real xi, type_real gamma);

std::pair<type_real, bool> get_local_edge_coordinate(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &global,
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim2> *,
                       Kokkos::HostSpace> &coorg,
    const specfem::mesh_entity::dim2::type &mesh_entity, type_real coord);

// Core locate_point logic that can be tested with raw data arrays
specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>
locate_point_core(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        &global_coordinates,
    const Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &index_mapping,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int ngnod, const int ngllx);

template <typename GraphType>
specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>
locate_point_core(const GraphType &graph,
                  const specfem::point::global_coordinates<
                      specfem::element::dimension_tag::dim2> &coordinates,
                  const Kokkos::View<type_real ****, Kokkos::LayoutRight,
                                     Kokkos::HostSpace> &global_coordinates,
                  const Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                     Kokkos::HostSpace> &control_node_coord,
                  const int ngnod);

specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>
locate_point_from_best_candidates(
    const std::vector<int> &best_candidates,
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coord,
    const int ngnod);

// 3D overloads - using different input types for overload resolution

// Using the 3D coordinate layout: (nspec, iz, iy, ix, icoord)
using MeshHostCoordinatesViewType3D =
    Kokkos::View<type_real *****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

std::tuple<int, int, int, int>
rough_location(const specfem::point::global_coordinates<
                   specfem::element::dimension_tag::dim3> &global,
               const MeshHostCoordinatesViewType3D coord);

std::vector<int> get_best_candidates(
    const int ispec_guess,
    const Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        index_mapping);

std::tuple<type_real, type_real, type_real> get_local_coordinates(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &global,
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim3> *,
                       Kokkos::HostSpace> &coorg,
    type_real xi, type_real eta, type_real gamma);

// Core locate_point logic that can be tested with raw data arrays
specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
locate_point_core(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const MeshHostCoordinatesViewType3D &global_coordinates,
    const Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &index_mapping,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
        &control_node_coordinates,
    const int ngnod, const int ngllx);

// Public API (previously in specfem::algorithms)

specfem::point::local_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

specfem::point::local_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type
        &team_member,
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

std::pair<type_real, bool> locate_point_on_edge(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);

specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point_on_edge(
    const type_real &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);

} // namespace specfem::algorithms::locate_point_impl
