#ifndef _ALGORITHMS_LOCATE_POINT_HPP
#define _ALGORITHMS_LOCATE_POINT_HPP

#include "compute/compute_mesh.hpp"
#include "point/coordinates.hpp"

namespace specfem {
namespace algorithms {

specfem::point::lcoord2 locate_point(const specfem::point::gcoord2 &coordinates,
                                     const specfem::compute::mesh &mesh);

specfem::point::gcoord2 locate_point(const specfem::point::lcoord2 &coordinates,
                                     const specfem::compute::mesh &mesh);

specfem::point::gcoord2
locate_point(const specfem::kokkos::HostTeam::member_type &team_member,
             const specfem::point::lcoord2 &coordinates,
             const specfem::compute::mesh &mesh);

} // namespace algorithms
} // namespace specfem

#endif
