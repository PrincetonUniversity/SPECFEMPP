#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Stress tensor at a quadrature point
 *
 * @tparam DimensionType Dimension of the element (2D or 3D)
 * @tparam MediumTag Medium tag for the element
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress {
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionType, MediumTag>::dimension();
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components();
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

  using ViewType =
      specfem::datatype::VectorPointViewType<type_real, dimension, components,
                                             UseSIMD>; ///< Underlying view type
                                                       ///< to store the stress
                                                       ///< tensor
  ///@}

  ViewType T; ///< View to store the stress tensor

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION stress() = default;

  /**
   * @brief Constructor
   *
   * @param T stress tensor
   */
  KOKKOS_FUNCTION stress(const ViewType &T) : T(T) {}
  ///@}

  /**
   * @brief Compute the product of the stress tensor with spatial derivatives
   *
   * /f$ F_{ij} = \sum_{k=1}^{N} T_{ik} \partial_k xi_j /f$
   *
   * @param partial_derivatives Spatial derivatives
   * @return ViewType Result of the product
   */
  KOKKOS_INLINE_FUNCTION
  ViewType operator*(const specfem::point::partial_derivatives<
                     specfem::dimension::type::dim2, false, UseSIMD>
                         &partial_derivatives) const {
    ViewType F;

    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(0, icomponent) = T(0, icomponent) * partial_derivatives.xix +
                         T(1, icomponent) * partial_derivatives.xiz;
      F(1, icomponent) = T(0, icomponent) * partial_derivatives.gammax +
                         T(1, icomponent) * partial_derivatives.gammaz;
    }

    return F;
  }
};
} // namespace point
} // namespace specfem
