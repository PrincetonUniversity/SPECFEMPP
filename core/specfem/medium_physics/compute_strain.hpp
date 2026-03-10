#pragma once

#include "specfem/data_access.hpp"
#include "specfem/medium/dim2/elastic/isotropic/strain.hpp"
#include "specfem/medium/dim3/elastic/isotropic/strain.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

// clang-format off
/**
 * @brief Compute strain tensor from field derivatives (kinematic, no material
 * properties needed).
 *
 * Generic strain computation interface that dispatches to medium-specific
 * implementations based on dimension and medium type.
 * Provides compile-time type safety through static assertions.
 *
 * @tparam PointFieldDerivativesType Point-wise displacement derivatives container
 * @param field_derivatives Displacement field derivatives at point
 * @return Strain tensor computed using medium-specific kinematics
 *
 * @code{.cpp}
 * using FieldDerivatives = specfem::point::field_derivatives<dim2, elastic_psv, false>;
 * FieldDerivatives derivs = ...; // Initialize field derivatives
 * auto strain = specfem::medium_physics::compute_strain(derivs);
 * @endcode
 */
// clang-format on
template <typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto
compute_strain(const PointFieldDerivativesType &field_derivatives) -> decltype(
    specfem::medium_physics::impl_compute_strain(field_derivatives)) {

  static_assert(
      specfem::data_access::is_point<PointFieldDerivativesType>::value &&
          specfem::data_access::is_field_derivatives<
              PointFieldDerivativesType>::value,
      "field_derivatives is not a point field derivatives type");

  return specfem::medium_physics::impl_compute_strain(field_derivatives);
}

// clang-format off
/**
 * @brief Compute deviatoric strain tensor from field derivatives.
 *
 * Subtracts trace/3 from the diagonal of the symmetric strain tensor.
 * For 2D plane-strain the stored deviatoric trace equals trace/3;
 * for 3D the deviatoric trace is exactly zero.
 *
 * @tparam PointFieldDerivativesType Point-wise displacement derivatives container
 * @param field_derivatives Displacement field derivatives at point
 * @return Deviatoric strain tensor
 */
// clang-format on
template <typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto
compute_deviatoric_strain(const PointFieldDerivativesType &field_derivatives)
    -> decltype(specfem::medium_physics::impl_compute_deviatoric_strain(
        field_derivatives)) {

  static_assert(
      specfem::data_access::is_point<PointFieldDerivativesType>::value &&
          specfem::data_access::is_field_derivatives<
              PointFieldDerivativesType>::value,
      "field_derivatives is not a point field derivatives type");

  return specfem::medium_physics::impl_compute_deviatoric_strain(
      field_derivatives);
}

} // namespace medium_physics
} // namespace specfem
