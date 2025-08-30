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

type_real locate_point_on_edge(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const int &ispec, const specfem::mesh_entity::type &constraint);

} // namespace algorithms
} // namespace specfem
