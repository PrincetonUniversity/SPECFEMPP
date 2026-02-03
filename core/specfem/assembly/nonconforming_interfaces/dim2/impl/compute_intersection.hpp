#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/mesh_entities.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"

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
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature);

/**
 * @brief Populates the transfer function for a given intersection.
 * The transfer function is a linear map from the element edge function basis to
 * the mortar basis.
 *
 * @param element1 - global coordinate representation of the first element.
 * @param element2 - global coordinate representation of the second element.
 * @param edge1 - edge on the first element
 * @param edge2 - edge on the second element
 * @param mortar_quadrature - a list of local (mortar space) knots describing
 *             　　　　　　　　　the quadrature rule on the interval [-1,1]
 * @param transfer_function1 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge1 to the mortar.
 * @param transfer_function2 - the nquad_mortar x ngll transfer tensor to
 * populate, mapping fields on edge2 to the mortar.
 */
template <typename TransferView1, typename TransferView2>
void set_transfer_functions(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    TransferView1 &transfer_function1, TransferView2 &transfer_function2);

/**
 * @brief Populates the transfer function for a given intersection.
 * The transfer function is a linear map from the element edge function basis to
 * the mortar basis.
 *
 * @param element1 - global coordinate representation of the first element.
 * @param element2 - global coordinate representation of the second element.
 * @param edge1 - edge on the first element
 * @param edge2 - edge on the second element
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
template <typename TransferView1, typename TransferView2,
          typename TransferView3, typename TransferView4>
void set_transfer_functions(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element1,
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &element2,
    const specfem::mesh_entity::dim2::type &edge1,
    const specfem::mesh_entity::dim2::type &edge2,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &mortar_quadrature,
    const Kokkos::View<type_real *, Kokkos::HostSpace> &element_quadrature,
    TransferView1 &transfer_function1, TransferView2 &transfer_function1_prime,
    TransferView3 &transfer_function2, TransferView4 &transfer_function2_prime);

} // namespace specfem::assembly::nonconforming_interfaces_impl
