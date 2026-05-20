#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <utility>
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

std::pair<std::pair<type_real, type_real>, bool> get_local_face_coordinate(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &global,
    const Kokkos::View<specfem::point::global_coordinates<
                           specfem::element::dimension_tag::dim3> *,
                       Kokkos::HostSpace> &coorg,
    const specfem::mesh_entity::dim3::type &mesh_entity,
    std::pair<type_real, type_real> coord);

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

// locate on edge (2D) global -> local
std::pair<type_real, bool> locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);

// locate on edge (2D) local -> global (edge coordinate)
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const type_real &coordinate,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::dim2::type &constraint);

// locate on edge (2D) local -> global (element coordinates + is on edge check)
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh_entity::dim2::type &constraint);

// locate on face (3D) global -> local

/**
 * @brief Given a face (ispec, constraint), finds the best fit local coordinate
 * on that face to the given global coordinates. Coordinates will be clamped to
 * [-1,1], even if a point outside that range is a better fit. In such a case,
 * the second return value will be false.
 *
 * @param coordinates - global coordinates to match to
 * @param mesh - assembly::mesh struct
 * @param ispec - element index whose local coordinates to find
 * @param constraint - face to compute for
 * @return std::pair<facial_coordinate_type,bool> - the face local coordinate
 * and whether or not the minimum found is a critical point (false is returned
 * if the best fit coordinate is out of bounds).
 */
std::pair<std::pair<type_real, type_real>, bool> locate_point(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint);

// locate on face (3D) local -> global

/**
 * @brief Convert face coordinate to global coordinates
 *
 * Given a face (ispec, constraint) and the coordinates along it, finds
 * the global coordinates.
 *
 * @param coordinates Local coordinate along face
 * @param mesh 2D spectral element mesh
 * @param ispec Element index whose local coordinates to find
 * @param constraint Edge to compute for
 * @return Global coordinates of the point
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const std::pair<type_real, type_real> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const int &ispec,
    const specfem::mesh_entity::type<specfem::element::dimension_tag::dim3>
        &constraint);

// locate on face (3D) local -> global (element coordinates + is on face check)
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
locate_point(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim3> &coordinates,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh_entity::dim3::type &constraint);
} // namespace specfem::algorithms::locate_point_impl
