// #pragma once

// #include "kokkos_abstractions.h"
// #include "specfem/point.hpp"
// #include "specfem_setup.hpp"

// namespace specfem {
// /**
//  * Jacobian namespace contains overloaded functions for serial (without
//  Kokkos)
//  * and Kokkos implementations (using team policy)
//  *
//  */
// namespace jacobian {

// using CoordTuple3D = std::tuple<type_real, type_real, type_real>;

// using MatrixTuple3D =
//     std::tuple<type_real, type_real, type_real, type_real, type_real,
//     type_real,
//                type_real, type_real, type_real>;
// /**
//  * @brief Compute global locations (x,z) from \f$(\xi, \gamma)\f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return CoordTuple3D (x,z) value for the point
//  */
// CoordTuple3D
// compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
//                   const HostScratchCoord &s_coorg, const int ngnod,
//                   const type_real xi, const type_real eta,
//                   const type_real gamma);

// /**
//  * @brief Compute global locations (x,z) from \f$(\xi, \gamma)\f$
//  *
//  *
//  * @param s_coorg View of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return CoordTuple3D (x,z) value for the point
//  */
// CoordTuple3D compute_locations(const HostCoord &s_coorg, const int ngnod,
//                                const type_real xi, const type_real eta,
//                                const type_real gamma);

// /**
//  * @brief Compute global locations (x,z) from shape function matrix
//  calcualted
//  * at \f$ (\xi, \gamma) \f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param shape3D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
//  * @return CoordTuple3D (x,z) value for the point
//  */
// CoordTuple3D
// compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
//                   const HostScratchCoord &s_coorg, const int ngnod,
//                   const std::vector<type_real> &shape3D);

// /**
//  * @brief Compute global locations (x,z) from shape function matrix
//  calcualted
//  * at  \f$ (\xi, \gamma) \f$
//  *
//  * @param coorg Global control node locations (x_a, z_a)
//  * @param ngnod Total number of control nodes per element
//  * @param shape3D shape function matrix calculated at  \f$ (\xi, \gamma) \f$
//  * @return CoordTuple3D (x,z) value for the point
//  */
// CoordTuple3D compute_locations(const HostCoord &s_coorg, const int ngnod,
//                                const std::vector<type_real> &shape3D);

// /**
//  * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return MatrixTuple3D partial
//  * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
//  * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
//  */
// MatrixTuple3D compute_jacobian_matrix(
//     const specfem::kokkos::HostTeam::member_type &teamMember,
//     const HostScratchCoord &s_coorg, const int ngnod, const type_real xi,
//     const type_real eta, const type_real gamma);

// /**
//  * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param dershape3D derivative of shape function matrix calculated at (xi,
//  * gamma)
//  * @return MatrixTuple3D partial
//  * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
//  * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
//  */
// MatrixTuple3D compute_jacobian_matrix(
//     const specfem::kokkos::HostTeam::member_type &teamMember,
//     const HostScratchCoord &s_coorg, const int ngnod,
//     const HostCoord &dershape3D);

// /**
//  * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg View of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return MatrixTuple3D partial
//  * derivatives \f$ (\partial x/\partial \xi, \partial x/\partial \gamma,
//  * \partial z/\partial \xi, \partial z/\partial \gamma) \f$
//  */
// MatrixTuple3D compute_jacobian_matrix(const HostCoord &s_coorg, const int
// ngnod,
//                                       const type_real xi, const type_real
//                                       eta, const type_real gamma);

// /**
//  * @brief compute jacobian given Jacobian matrix at a point
//  *
//  * @param xxi Jacobian matrix \f$ \partial x/\partial \xi \f$
//  * @param zxi Jacobian matrix \f$ \partial z/\partial \xi \f$
//  * @param xgamma Jacobian matrix \f$ \partial x/\partial \gamma \f$
//  * @param zgamma Jacobian matrix \f$ \partial z/\partial \gamma \f$
//  * @return type_real computed jacobian
//  */
// type_real compute_jacobian(const type_real xxi, const type_real zxi,
//                            const type_real xgamma, const type_real zgamma);

// /**
//  * @brief compute jacobian at a particular point
//  *
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return type_real computed jacobian
//  */
// type_real
// compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
//                  const HostScratchCoord &s_coorg, const int ngnod,
//                  const type_real xi, const type_real eta,
//                  const type_real gamma);

// /**
//  * @brief compute jacobian at a particular point using derivatives of shape
//  * function
//  *
//  * @param s_coorg scratch view of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param dershape3D derivative of shape function matrix calculated at \f$
//  (\xi,
//  * \gamma) \f$
//  * @return type_real computed jacobian
//  */
// type_real
// compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
//                  const HostScratchCoord &s_coorg, const int ngnod,
//                  const HostCoord &dershape3D);

// /**
//  * @brief Compute Jacobian matrix at  \f$ (\xi, \gamma) \f$
//  *
//  * @note This function can only be called within a team policy
//  *
//  * @param teamMember Kokkos team policy team member
//  * @param s_coorg View of coorg subviewed at required element
//  * @param ngnod Total number of control nodes per element
//  * @param xi \f$ \xi \f$ value of the point
//  * @param eta \f$ \eta \f$ value of the point
//  * @param gamma \f$ \gamma \f$ value of point
//  * @return MatrixTuple3D partial
//  * derivatives \f$ (\partial \xi/ \partial x, \partial \gamma/ \partial x,
//  * \partial \xi/ \partial z, \partial \gamma/ \partial z) \f$
//  */
// MatrixTuple3D compute_inverted_derivatives(const HostCoord &s_coorg,
//                                            const int ngnod, const type_real
//                                            xi, const type_real eta, const
//                                            type_real gamma);

// specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
// compute_derivatives(const specfem::kokkos::HostTeam::member_type &teamMember,
//                     const HostScratchCoord &s_coorg, const int ngnod,
//                     const HostCoord &dershape3D);

// } // namespace jacobian
// } // namespace specfem
