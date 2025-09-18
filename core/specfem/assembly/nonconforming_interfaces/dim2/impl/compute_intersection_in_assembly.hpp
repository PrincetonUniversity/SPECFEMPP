#pragma once

#include "quadrature/interface.hpp"
#include "quadrature/quadrature.hpp"
#include "specfem/assembly.hpp"

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
 * @param mesh - the assembly::mesh object
 * @param edge - the adjacency graph edge of the intersection.
 *               mesh.graph()[edge] should reference the edge we want to
 *               construct the intersection between.
 * @param mortar_quadrature - a list of local (mortar space) knots describing
 *             　　　　　　　　　the quadrature rule on the interval [-1,1]
 * @return std::vector<std::pair<type_real, type_real> > - a vector of length
 *      mortar_quadrature.extent(0), with each element retval[i] being a pair
 *      (element1_local_coord, element2_local_coord) corresponding to the mortar
 *      knot mortar_quadrature(i).
 */
template <typename EdgeType>
std::vector<std::pair<type_real, type_real> > compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge, const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature);
} // namespace specfem::assembly::nonconforming_interfaces_impl
