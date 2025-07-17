#ifndef _ALGORITHMS_LOCATE_POINT_HPP
#define _ALGORITHMS_LOCATE_POINT_HPP

#include "compute/compute_mesh.hpp"
#include "specfem/point.hpp"

namespace specfem {
namespace algorithms {

specfem::point::local_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::compute::mesh &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::compute::mesh &mesh);

specfem::point::global_coordinates<specfem::dimension::type::dim2> locate_point(
    const specfem::kokkos::HostTeam::member_type &team_member,
    const specfem::point::local_coordinates<specfem::dimension::type::dim2>
        &coordinates,
    const specfem::compute::mesh &mesh);

} // namespace algorithms
} // namespace specfem

#endif
