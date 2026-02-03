#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * @brief Struct to store the assembled index for a quadrature point
 *
 * @tparam using_simd Flag to indicate if this is a simd index
 */
template <specfem::element::dimension_tag DimensionTag> struct weights;

/**
 * @brief Struct to store the assembled index for a quadrature point
 *
 * This struct stores a 1D index that corresponds to a global numbering of the
 * quadrature point within the mesh.
 *
 */
template <>
struct weights<specfem::element::dimension_tag::dim2>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::weights,
          specfem::element::dimension_tag::dim2, false> {
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim2; ///< Dimension tag
  type_real wz; ///< Weight of the quadrature point in the z direction within
                ///< the spectral element
  type_real wx; ///< Weight of the quadrature point in the x direction within
                ///< the spectral element

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights() = default;

  /**
   * @brief Constructor with values
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights(const type_real &wz, const type_real &wx) : wz(wz), wx(wx) {}
  ///@}

  /**
   * @brief Get the product of the weights
   *
   * @return type_real Product of the weights
   */
  KOKKOS_INLINE_FUNCTION type_real product() const { return wz * wx; }
};

template <>
struct weights<specfem::element::dimension_tag::dim3>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::weights,
          specfem::element::dimension_tag::dim3, false> {
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag
  type_real wz; ///< Weight of the quadrature point in the z direction within
                ///< the spectral element
  type_real wy; ///< Weight of the quadrature point in the y direction within
                ///< the spectral element
  type_real wx; ///< Weight of the quadrature point in the x direction within
                ///< the spectral element

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights() = default;

  /**
   * @brief Constructor with values
   *
   */
  KOKKOS_INLINE_FUNCTION
  weights(const type_real &wz, const type_real &wy, const type_real &wx)
      : wz(wz), wy(wy), wx(wx) {}
  ///@}

  /**
   * @brief Get the product of the weights
   *
   * @return type_real Product of the weights
   */
  KOKKOS_INLINE_FUNCTION type_real product() const { return wz * wy * wx; }
};

} // namespace specfem::point
