#pragma once

#include "enumerations/mesh_entities.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"

namespace specfem {
namespace algorithms {

specfem::point::local_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh);

specfem::point::local_coordinates<specfem::dimension::type::dim3> locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim3> locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim3> locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim3>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim3> &mesh);

/**
 * @brief Given an edge (ispec, constraint), finds the best fit local coordinate
 * on that edge to the given global coordinates. Coordinates will be clamped to
 * [-1,1], even if a point outside that range is a better fit. In such a case,
 * the second return value will be false.
 *
 * @param coordinates - global coordinates to match to
 * @param mesh - assembly::mesh struct
 * @param ispec - element index whose local coordinates to find
 * @param constraint - edge to compute for
 * @return std::pair<type_real,bool> - the edge local coordinate and whether or
 * not the minimum found is a critical point (false is returned if the best fit
 * coordinate is out of bounds).
 */
std::pair<type_real, bool> locate_point_on_edge(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::type &constraint);
/**
 * @brief Given an edge (ispec, constraint) and the coordinate along it, finds
 * the global coordinates.
 *
 * @param coordinate - local coordinate along edge
 * @param mesh - assembly::mesh struct
 * @param ispec - element index whose local coordinates to find
 * @param constraint - edge to compute for
 * @return specfem::point::global_coordinates<specfem::dimension::type::dim2> -
 * the global coordinates of the point.
 */
specfem::point::global_coordinates<specfem::dimension::type::dim2>
locate_point_on_edge(
    const type_real &coordinate,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::type &constraint);

} // namespace algorithms
} // namespace specfem
