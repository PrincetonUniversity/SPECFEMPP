#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file stacey.hpp
 * @brief Stacey absorbing boundary conditions
 *
 */

namespace specfem {
namespace boundary_conditions {

using stacey_type =
    std::integral_constant<specfem::element::boundary_tag,
                           specfem::element::boundary_tag::stacey>;

namespace impl {

/**
 * @brief Apply Stacey absorbing boundary conditions for Elastic P-SV medium
 *
 * The traction vector is computed based on P-SV separation:
 * \f[
 * \mathbf{t} = - \rho V_p (\mathbf{v} \cdot \mathbf{n}) \mathbf{n} - \rho V_s
 * (\mathbf{v} - (\mathbf{v} \cdot \mathbf{n}) \mathbf{n})
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointVelocityType Point velocity type
 * @tparam ViewType View type
 * @param boundary Boundary object
 * @param property Property object
 * @param velocity Velocity object
 * @param traction Traction object to be updated
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_psv_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_psv_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

/**
 * @brief Apply Stacey absorbing boundary conditions for Elastic SH medium
 *
 * For SH waves (scalar anti-plane shear), the traction is:
 * \f[
 * t = - \rho V_s v
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointVelocityType Point velocity type
 * @tparam ViewType View type
 * @param boundary Boundary object
 * @param property Property object
 * @param velocity Velocity object
 * @param traction Traction object to be updated
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_sh_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_sh_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

/**
 * @brief Apply Stacey absorbing boundary conditions for Elastic Cosserat
 * (P-SV-t) medium
 *
 * Updates traction for both translational and rotational components.
 *
 * Translational part (P-SV):
 * \f[
 * \mathbf{t}_{trans} = - \rho V_p (\mathbf{v}_{trans} \cdot \mathbf{n})
 * \mathbf{n} - \rho V_s (\mathbf{v}_{trans} - (\mathbf{v}_{trans} \cdot
 * \mathbf{n}) \mathbf{n})
 * \f]
 *
 * Rotational part (Twist):
 * \f[
 * t_{rot} = - J V_t \omega
 * \f]
 * where \f$J\f$ is the micro-inertia (polar moment of inertia times density)
 * and \f$V_t\f$ is the twisting velocity.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointVelocityType Point velocity type
 * @tparam ViewType View type
 * @param boundary Boundary object
 * @param property Property object
 * @param velocity Velocity object
 * @param traction Traction object to be updated
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_psv_t_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_base_elastic_psv_t_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

/**
 * @brief Apply Stacey absorbing boundary conditions for Acoustic Isotropic
 * medium
 *
 * The boundary condition applied to the potential (or pressure) field is:
 * \f[
 * t = - \frac{1}{\rho V_p} v
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointVelocityType Point velocity type
 * @tparam ViewType View type
 * @param boundary Boundary object
 * @param property Property object
 * @param velocity Velocity object
 * @param traction Traction object to be updated
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_enforce_traction(
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic> &,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic> &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_enforce_traction(
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic> &,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic> &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction);

} // namespace impl

/**
 * @brief Apply Stacey absorbing boundary conditions
 *
 * Implements the first-order paraxial absorbing boundary condition (Stacey
 * condition). The traction vector \f$\mathbf{t}\f$ is computed to absorb
 * outgoing waves based on the velocity field \f$\mathbf{v}\f$ and the medium
 * properties.
 *
 * The boundary condition is defined as:
 * \f[
 * \mathbf{t} = - \rho V_p (\mathbf{v} \cdot \mathbf{n}) \mathbf{n} - \rho V_s
 * (\mathbf{v} - (\mathbf{v} \cdot \mathbf{n}) \mathbf{n})
 * \f]
 *
 * where:
 * - \f$\rho\f$ is the density,
 * - \f$V_p\f$ is the P-wave velocity,
 * - \f$V_s\f$ is the S-wave velocity,
 * - \f$\mathbf{n}\f$ is the outward unit normal vector to the boundary.
 *
 * This effectively separates the wavefield into normal (P-wave) and tangential
 * (S-wave) components and applies the corresponding impedance to absorb them.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param property Property object
 * @param field Field object
 * @param acceleration Acceleration object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration);

/**
 * @brief Compute mass matrix terms for Stacey absorbing boundary conditions
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointMassMatrixType Point mass matrix type
 * @param dt Time step
 * @param boundary Boundary object
 * @param property Property object
 * @param mass_matrix Mass matrix object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FUNCTION void impl_compute_mass_matrix_terms(
    const stacey_type &, const type_real dt, const PointBoundaryType &boundary,
    const PointPropertyType &property, PointMassMatrixType &mass_matrix);

} // namespace boundary_conditions
} // namespace specfem
