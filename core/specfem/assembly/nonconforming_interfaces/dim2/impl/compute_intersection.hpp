#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "quadrature/interface.hpp"
#include "quadrature/quadrature.hpp"
#include "specfem/point.hpp"

#include <vector>

namespace specfem::assembly::nonconforming_interfaces_impl {

/**
 * @brief Computes the intersection between two elements, returning the knots of
 * the "mortar", the codimension 1 element joining them at their intersection,
 * in the local coordinates of both elements.
 *
 * The intersection is expected to span the space between two of the four
 * candidate endpoints (endpoints of either element edge).
 *
 * @param element1 - global coordinate representation of the first element.
 * @param element2 - global coordinate representation of the second element.
 * @param edge1 - edge on the first element
 * @param edge2 - edge on the second element
 * @param mortar_quadrature - a list of local (mortar space) knots describing
 *             　　　　　　　　　the quadrature rule on the interval [-1,1]
 * @return std::vector<std::pair<type_real, type_real> > - a vector of length
 *      mortar_quadrature.extent(0), with each element retval[i] being a pair
 *      (element1_local_coord, element2_local_coord) corresponding to the mortar
 *      knot mortar_quadrature(i).
 */
std::vector<std::pair<type_real, type_real> > compute_intersection(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::type &edge1,
    const specfem::mesh_entity::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature);

} // namespace specfem::assembly::nonconforming_interfaces_impl
