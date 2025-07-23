#pragma once

#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {

/**
 * Jacobian namespace contains overloaded functions for serial (without Kokkos)
 * and Kokkos implementations (using team policy)
 *
 */
namespace jacobian {

/**
 * @brief Compute global locations (x,z) from shape function matrix calcualted
 * at  \f$ (\xi, \gamma) \f$
 *
 * @param coorg Global control node locations (x_a, z_a)
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @param shape2D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
 * @return std::tuple<type_real, type_real> (x,z) value for the point
 */
std::tuple<type_real, type_real>
compute_locations(const specfem::kokkos::HostView2d<type_real> &s_coorg,
                  const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @param dershape2D derivative of shape function matrix calculated at (xi,
 * gamma)
 * @return std::tuple<type_real, type_real, type_real, type_real, type_real>
 * partial derivatives \f$ (\partial \xi/ \partial x, \partial \gamma/ \partial
 * x,
 * \partial \xi/ \partial z, \partial \gamma/ \partial z) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_derivatives(const specfem::kokkos::HostView2d<type_real> &s_coorg,
                    const int ngnod, const type_real xi, const type_real gamma);

std::tuple<type_real, type_real, type_real, type_real, type_real>
compute_derivatives(const specfem::kokkos::HostView2d<type_real> &s_coorg,
                    const int ngnod,
                    const std::vector<std::vector<type_real> > &dershape2D);

} // namespace jacobian
} // namespace specfem
