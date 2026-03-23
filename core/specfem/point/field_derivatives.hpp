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
};

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
} // namespace specfem::point
