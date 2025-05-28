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
  constexpr static auto dimension = DimensionType;
  constexpr static int ndim =
      specfem::element::attributes<DimensionType, MediumTag>::dimension;
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static int components =
      specfem::element::attributes<DimensionType, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type

  using ViewType =
      specfem::datatype::VectorPointViewType<type_real, components, ndim,
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
   * Following Komatitsch and Tromp (1999), the product of the stress tensor
   * with the spatial derivatives is computed as follows:
   *
   * /f{equation}{ F_{ik} = \sum_{k=1}^{N} T_{ij} \partial_j xi_k /f}
   *
   * In detail, the stress integrand array is intuitively defined as
   *
   * \f{equation}{ F_{ik} = F_{x_i, \xi_k} = F(i, k),/f}
   *
   * where \f$x_i = [x,z]\f$, and \f$ \xi_k = [\xi, \gamma] \f$.
   *
   * The stress integrand is the populated as follows:
   *
   * \f{equation}{ F(i, k) = T(i, j) \partial xi_k / \partial \x_j \f}
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
      // F(i, k) = T(i, 0) * \partial \xi_k / \partial x
      //         + T(i, 1) * \partial \xi_k / \partial z

      // F(0, 0) = F_{x,\xi} = sigma_xx * xix + sigma_xz * xiz
      // F(1, 0) = F_{z,\xi} = sigma_zx * xix + sigma_zz * xiz
      // ...
      F(icomponent, 0) = T(icomponent, 0) * partial_derivatives.xix +
                         T(icomponent, 1) * partial_derivatives.xiz;

      // F(0, 1) = F_{z,\gamma} = sigma_xz * gammax + sigma_zz * gammaz
      // F(1, 1) = F_{x,\gamma} = sigma_xx * gammax + sigma_xz * gammaz
      // ...
      F(icomponent, 1) = T(icomponent, 0) * partial_derivatives.gammax +
                         T(icomponent, 1) * partial_derivatives.gammaz;
    }

    return F;
  }
};
} // namespace point
} // namespace specfem
