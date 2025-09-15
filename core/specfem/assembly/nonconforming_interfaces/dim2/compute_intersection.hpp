#pragma once

#include "quadrature/interface.hpp"
#include "quadrature/quadrature.hpp"
#include "specfem/assembly.hpp"

#include <vector>

namespace specfem::assembly {
namespace nonconforming_interfaces {

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
std::vector<std::pair<type_real, type_real> > compute_intersection(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const boost::graph_traits<specfem::mesh::adjacency_graph<
        specfem::dimension::type::dim2>::GraphType>::edge_descriptor &edge,
    const Kokkos::View<type_real *> &mortar_quadrature);
} // namespace nonconforming_interfaces
} // namespace specfem::assembly
