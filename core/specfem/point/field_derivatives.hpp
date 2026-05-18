#pragma once

#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {
namespace impl {

/// @private Implementation detail
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct field_derivatives
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::field_derivatives, DimensionTag,
          UseSIMD> {

private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::field_derivatives, DimensionTag,
      UseSIMD>; ///< Base type of the
                ///< point field
                ///< derivatives
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  static constexpr int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  constexpr static auto medium_tag = MediumTag; ///< Medium tag for the element
  constexpr static int num_dimensions =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               num_dimensions>;
  ///@}

  value_type du; ///< View to store the field derivatives.

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION field_derivatives() = default;

  /**
   * @brief Constructor
   *
   * @param du Field derivatives
   */
  KOKKOS_FUNCTION field_derivatives(const value_type &du) : du(du) {}
  ///@}

  /**
   * @name Arithmetic operators
   */
  ///@{

  /// Element-wise addition: (a + b).du(i,k) = a.du(i,k) + b.du(i,k)
  KOKKOS_INLINE_FUNCTION field_derivatives
  operator+(const field_derivatives &rhs) const {
    field_derivatives result;
    for (int i = 0; i < components; ++i)
      for (int k = 0; k < num_dimensions; ++k)
        result.du(i, k) = du(i, k) + rhs.du(i, k);
    return result;
  }

  /// Scalar multiplication (right): (a * s).du(i,k) = a.du(i,k) * s
  KOKKOS_INLINE_FUNCTION field_derivatives
  operator*(const type_real &rhs) const {
    field_derivatives result;
    for (int i = 0; i < components; ++i)
      for (int k = 0; k < num_dimensions; ++k)
        result.du(i, k) = du(i, k) * rhs;
    return result;
  }
  ///@}
};

/// Scalar multiplication (left): (s * a).du(i,k) = s * a.du(i,k)
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
KOKKOS_INLINE_FUNCTION field_derivatives<DimensionTag, MediumTag, UseSIMD>
operator*(const type_real &lhs,
          const field_derivatives<DimensionTag, MediumTag, UseSIMD> &rhs) {
  return rhs * lhs;
}

} // namespace impl

/**
 * @brief Store field derivatives for a quadrature point
 *
 * The field derivatives are given by:
 * \f$ du_{i,k} = \partial_i u_k \f$
 *
 * @tparam Tags The tags for the element where the quadrature point is located
 */
template <typename Tags>
using field_derivatives =
    impl::field_derivatives<Tags::dimension_tag, Tags::medium_tag,
                            Tags::using_simd>;

/**
 * @brief Sentinel type returned by FieldDerivativesPack::get_dv() when
 * there is no velocity gradient (attenuation_tag::none).
 *
 * Carries a compile-time flag so that load/store helpers can detect it via
 * trait without a cross-include dependency.
 */
struct null_field_derivatives {
  static constexpr bool is_null_field_derivatives_v = true;
};

} // namespace specfem::point
