#pragma once

#include "quadrature/interface.hpp"
#include "quadrature/quadrature.hpp"
#include "specfem/assembly.hpp"

#include <vector>

namespace specfem::assembly::nonconforming_interfaces_impl {

/**
 * @brief A helper function that retrieves the element global coordinates and
 * edge orientation of a given adjacency in the adjacency graph.
 *
 * @param mesh - the assembly::mesh object
 * @param edge - the adjacency graph edge of the intersection.
 *               mesh.graph()[edge] should reference the edge we want to
 *               construct the intersection between.
 * @return std::tuple<
 * Kokkos::View<
 * specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
 * Kokkos::HostSpace>,
 * Kokkos::View<
 * specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
 * Kokkos::HostSpace>,
 * specfem::mesh_entity::type, specfem::mesh_entity::type> - in order: the
 * cooordinates of the source element, the coordinates of the target element,
 * the orientation on the source element in the intersection, and the
 * orientation on the target element in the intersection.
 */
template <typename EdgeType>
std::tuple<
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>,
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>,
    specfem::mesh_entity::type, specfem::mesh_entity::type>
expand_edge_index(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge);

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
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature);
} // namespace specfem::assembly::nonconforming_interfaces_impl

/**
 * @brief Populates the transfer function for a given intersection.
 * The transfer function is a linear map from the element edge function basis to
 * the mortar basis.
 *
 * @param mesh - the assembly::mesh object
 * @param edge - the adjacency graph edge of the intersection.
 *               mesh.graph()[edge] should reference the edge we want to
 *               construct the intersection between.
 * @param mortar_quadrature - a list of local (mortar space) knots describing
 *             　　　　　　　　　the quadrature rule on the interval [-1,1]
 * @param transfer_function1 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge1 to the mortar.
 * @param transfer_function2 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge2 to the mortar.
 */
template <typename EdgeType>
void set_transfer_functions(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2);

/**
 * @brief Populates the transfer function for a given intersection.
 * The transfer function is a linear map from the element edge function basis to
 * the mortar basis.
 *
 * @param mesh - the assembly::mesh object
 * @param edge - the adjacency graph edge of the intersection.
 *               mesh.graph()[edge] should reference the edge we want to
 *               construct the intersection between.
 * @param mortar_quadrature - a list of local (mortar space) knots describing
 *             　　　　　　　　　the quadrature rule on the interval [-1,1]
 * @param transfer_function1 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge1 to the mortar.
 * @param transfer_function1_prime - derivative of transfer_function1 w.r.t.
 * edge coordinate.
 * @param transfer_function2 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge2 to the mortar.
 * @param transfer_function2_prime - derivative of transfer_function2 w.r.t.
 * edge coordinate.
 */
template <typename EdgeType>
void set_transfer_functions(
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const EdgeType &edge,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function1_prime,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2,
    Kokkos::View<type_real **, Kokkos::HostSpace> &transfer_function2_prime);
