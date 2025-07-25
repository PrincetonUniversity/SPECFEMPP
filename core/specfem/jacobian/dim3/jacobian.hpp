#pragma once

#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

/**
 * Jacobian namespace contains overloaded functions for serial (without Kokkos)
 * and Kokkos implementations (using team policy)
 *
 */
namespace specfem::jacobian {

/**
 * @brief Compute global locations (x,z) from shape function matrix calcualted
 * at  \f$ (\xi, \gamma) \f$
 *
 * @param coorg Global control node locations (x_a, z_a)
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param gamma \f$ \gamma \f$ value of point
 * @param shape3D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
 * @return specfem::point::global_coordinates<specfem::dimension::type::dim3>
 * (x,y,z) value for the point
 */
specfem::point::global_coordinates<specfem::dimension::type::dim3>
compute_locations(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

/**
 * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param gamma \f$ \gamma \f$ value of point
 * @param dershape3D derivative of shape function matrix calculated at (xi,
 * eta, gamma)
    * @return std::tuple<type_real, type_real, type_real, type_real, type_real,
                            type_real, type_real, type_real, type_real> partial
 derivatives \f$ (\partial \xi/ \partial x, \partial \eta/ \partial
 * x,
 * \partial \xi/ \partial z, \partial \eta/ \partial z, \partial \gamma/
 \partial
 * z) \f$
 */
specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
compute_jacobian(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

} // namespace specfem::jacobian
