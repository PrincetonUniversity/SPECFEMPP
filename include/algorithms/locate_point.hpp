#pragma once

#include "specfem/assembly.hpp"
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

} // namespace algorithms
} // namespace specfem
