#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "partial_derivatives.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Stress tensor at a quadrature point
 *
 * @tparam DimensionTag Dimension of the element (2D or 3D)
 * @tparam MediumTag Medium tag for the element
 * @tparam UseSIMD Use SIMD instructions
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress
    : public specfem::accessor::Accessor<specfem::accessor::type::point,
                                         specfem::data_class::type::stress,
                                         DimensionTag, UseSIMD> {
private:
  using base_type =
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::stress,
                                  DimensionTag,
                                  UseSIMD>; ///< Base accessor type
public:
  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;
  ///@}

  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD type
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>; ///< Underlying view
                                                           ///< type to store
                                                           ///< the stress
                                                           ///< tensor
  ///@}

  value_type T; ///< View to store the stress tensor

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
  KOKKOS_FUNCTION stress(const value_type &T) : T(T) {}
  ///@}

  /**
   * @brief Compute the product of the stress tensor with spatial derivatives
   *
   * /f$ F_{ij} = \sum_{k=1}^{N} T_{ik} \partial_k xi_j /f$
   *
   * @param partial_derivatives Spatial derivatives
   * @return value_type Result of the product
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(const specfem::point::partial_derivatives<
                       specfem::dimension::type::dim2, true, UseSIMD>
                           &partial_derivatives) const {
    value_type F;

    // The correct expression for F does not include Jacobian factor here.
    // However, for non regular meshes, the expression A5 in (Komatitsch et. al.
    // 2005) results in numerical instabilities. This is because spatial
    // derivatives can be small leading to precision errors. Multiplying by the
    // Jacobian factor helps normalize the result. We then avoid the jacobian
    // factor when computing the divergence in equation (A6).
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(icomponent, 0) = partial_derivatives.jacobian *
                         (T(icomponent, 0) * partial_derivatives.xix +
                          T(icomponent, 1) * partial_derivatives.xiz);
      F(icomponent, 1) = partial_derivatives.jacobian *
                         (T(icomponent, 0) * partial_derivatives.gammax +
                          T(icomponent, 1) * partial_derivatives.gammaz);
    }

    return F;
  }
};
} // namespace point
} // namespace specfem
