#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Store field derivatives for a quadrature point
 *
 * The field derivatives are given by:
 * \f$ du_{i,k} = \partial_i u_k \f$
 *
 * @tparam DimensionTag The dimension of the element where the quadrature point
 * is located
 * @tparam MediumTag The medium of the element where the quadrature point is
 * located
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct field_derivatives
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::field_derivatives, DimensionTag,
          UseSIMD> {

private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
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

} // namespace point
} // namespace specfem
