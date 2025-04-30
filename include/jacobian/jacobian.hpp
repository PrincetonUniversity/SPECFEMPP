#ifndef _JACOBIAN_HPP
#define _JACOBIAN_HPP

#include "kokkos_abstractions.h"
#include "specfem/point/partial_derivatives.hpp"
#include "specfem_setup.hpp"

namespace specfem {
/**
 * Jacobian namespace contains overloaded functions for serial (without Kokkos)
 * and Kokkos implementations (using team policy)
 *
 */
namespace jacobian {

/**
 * @brief Compute global locations (x,z) from \f$(\xi, \gamma)\f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real> (x,z) value for the point
 */
std::tuple<type_real, type_real>
compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
                  const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
                  const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute global locations (x,z) from \f$(\xi, \gamma)\f$
 *
 *
 * @param s_coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real> (x,z) value for the point
 */
std::tuple<type_real, type_real>
compute_locations(const specfem::kokkos::HostView2d<type_real> s_coorg,
                  const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute global locations (x,z) from shape function matrix calcualted
 * at \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param shape2D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
 * @return std::tuple<type_real, type_real> (x,z) value for the point
 */
std::tuple<type_real, type_real>
compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
                  const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
                  const int ngnod, const std::vector<type_real> shape2D);

/**
 * @brief Compute global locations (x,z) from shape function matrix calcualted
 * at  \f$ (\xi, \gamma) \f$
 *
 * @param coorg Global control node locations (x_a, z_a)
 * @param ngnod Total number of control nodes per element
 * @param shape2D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
 * @return std::tuple<type_real, type_real> (x,z) value for the point
 */
std::tuple<type_real, type_real>
compute_locations(const specfem::kokkos::HostView2d<type_real> s_coorg,
                  const int ngnod, const std::vector<type_real> shape2D);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
 * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_partial_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param dershape2D derivative of shape function matrix calculated at (xi,
 * gamma)
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
 * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_partial_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
 * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_partial_derivatives(
    const specfem::kokkos::HostView2d<type_real> s_coorg, const int ngnod,
    const type_real xi, const type_real gamma);

/**
 * @brief compute jacobian given partial derivatives at a point
 *
 * @param xxi partial derivative \f$ \partial x/\partial \xi \f$
 * @param zxi partial derivative \f$ \partial z/\partial \xi \f$
 * @param xgamma partial derivative \f$ \partial x/\partial \gamma \f$
 * @param zgamma partial derivative \f$ \partial z/\partial \gamma \f$
 * @return type_real computed jacobian
 */
type_real compute_jacobian(const type_real xxi, const type_real zxi,
                           const type_real xgamma, const type_real zgamma);

/**
 * @brief compute jacobian at a particular point
 *
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return type_real computed jacobian
 */
type_real
compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
                 const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
                 const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief compute jacobian at a particular point using derivatives of shape
 * function
 *
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param dershape2D derivative of shape function matrix calculated at \f$ (\xi,
 * \gamma) \f$
 * @return type_real computed jacobian
 */
type_real
compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
                 const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
                 const int ngnod,
                 const specfem::kokkos::HostView2d<type_real> dershape2D);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial \xi/ \partial x, \partial \gamma/ \partial x,
 * \partial \xi/ \partial z, \partial \gamma/ \partial z) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg scratch view of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param dershape2D derivative of shape function matrix calculated at (xi,
 * gamma)
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial \xi/ \partial x, \partial \gamma/ \partial x,
 * \partial \xi/ \partial z, \partial \gamma/ \partial z) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D);

/**
 * @brief Compute partial derivatives at  \f$ (\xi, \gamma) \f$
 *
 * @note This function can only be called within a team policy
 *
 * @param teamMember Kokkos team policy team member
 * @param s_coorg View of coorg subviewed at required element
 * @param ngnod Total number of control nodes per element
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @return std::tuple<type_real, type_real, type_real, type_real> partial
 * derivatives \f$ (\partial \xi/ \partial x, \partial \gamma/ \partial x,
 * \partial \xi/ \partial z, \partial \gamma/ \partial z) \f$
 */
std::tuple<type_real, type_real, type_real, type_real>
compute_inverted_derivatives(
    const specfem::kokkos::HostView2d<type_real> s_coorg, const int ngnod,
    const type_real xi, const type_real gamma);

specfem::point::partial_derivatives<specfem::dimension::type::dim2, true, false>
compute_derivatives(const specfem::kokkos::HostTeam::member_type &teamMember,
                    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
                    const int ngnod,
                    const specfem::kokkos::HostView2d<type_real> dershape2D);

} // namespace jacobian
} // namespace specfem

#endif
