#pragma once

#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/point/field_derivatives.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_strain_computation_dim3_elastic_isotropic
 */

/**
 * @ingroup specfem_strain_computation_dim3_elastic_isotropic
 * @brief Compute symmetric strain tensor for 3D elastic media.
 *
 * @tparam FieldDerivativesType Point field-derivatives type (deduced)
 * @param field_derivatives Displacement gradients
 * @return 3×3 symmetric strain tensor
 */
template <typename FieldDerivativesType,
          std::enable_if_t<FieldDerivativesType::medium_tag ==
                               specfem::element::medium_tag::elastic,
                           int> = 0>
KOKKOS_INLINE_FUNCTION
    specfem::datatype::TensorPointViewType<type_real, 3, 3,
                                           FieldDerivativesType::using_simd>
    impl_compute_strain(const FieldDerivativesType &field_derivatives) {

  using datatype = typename specfem::datatype::simd<
      type_real, FieldDerivativesType::using_simd>::datatype;
  const auto &du = field_derivatives.du;

  specfem::datatype::TensorPointViewType<type_real, 3, 3,
                                         FieldDerivativesType::using_simd>
      epsilon;

  // Normal strains
  epsilon(0, 0) = du(0, 0); // ε_xx
  epsilon(1, 1) = du(1, 1); // ε_yy
  epsilon(2, 2) = du(2, 2); // ε_zz

  // Symmetric shear strains
  const datatype eps_xy = static_cast<type_real>(0.5) * (du(0, 1) + du(1, 0));
  const datatype eps_xz = static_cast<type_real>(0.5) * (du(0, 2) + du(2, 0));
  const datatype eps_yz = static_cast<type_real>(0.5) * (du(1, 2) + du(2, 1));

  epsilon(0, 1) = eps_xy;
  epsilon(1, 0) = eps_xy;
  epsilon(0, 2) = eps_xz;
  epsilon(2, 0) = eps_xz;
  epsilon(1, 2) = eps_yz;
  epsilon(2, 1) = eps_yz;

  return epsilon;
}

/**
 * @ingroup specfem_strain_computation_dim3_elastic_isotropic
 * @brief Compute deviatoric strain tensor for 3D elastic media.
 *
 * Subtracts trace/3 from diagonal. In 3D the deviatoric trace is exactly zero.
 *
 * @tparam FieldDerivativesType Point field-derivatives type (deduced)
 * @param field_derivatives Displacement gradients
 * @return 3×3 deviatoric strain tensor
 */
template <typename FieldDerivativesType,
          std::enable_if_t<FieldDerivativesType::medium_tag ==
                               specfem::element::medium_tag::elastic,
                           int> = 0>
KOKKOS_INLINE_FUNCTION specfem::datatype::TensorPointViewType<
    type_real, 3, 3, FieldDerivativesType::using_simd>
impl_compute_deviatoric_strain(const FieldDerivativesType &field_derivatives) {

  auto epsilon = impl_compute_strain(field_derivatives);

  const auto trace = epsilon(0, 0) + epsilon(1, 1) + epsilon(2, 2);
  const auto third_trace = static_cast<type_real>(1.0 / 3.0) * trace;

  epsilon(0, 0) = epsilon(0, 0) - third_trace;
  epsilon(1, 1) = epsilon(1, 1) - third_trace;
  epsilon(2, 2) = epsilon(2, 2) - third_trace;

  return epsilon;
}

/**
 * @ingroup specfem_strain_computation_dim3_elastic_isotropic
 * @brief Trace of a 3×3 elastic strain tensor (ε_xx + ε_yy + ε_zz).
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param tensor 3×3 strain tensor
 * @return Scalar trace value
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    typename specfem::datatype::simd<type_real, UseSIMD>::datatype
    impl_trace(const specfem::datatype::TensorPointViewType<type_real, 3, 3,
                                                            UseSIMD> &tensor) {
  return tensor(0, 0) + tensor(1, 1) + tensor(2, 2);
}

} // namespace medium_physics
} // namespace specfem
