#pragma once

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "jacobian_matrix.hpp"
#include "specfem/data_access.hpp"
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
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::stress, DimensionTag, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::stress, DimensionTag,
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

  constexpr static specfem::dimension::type dimension_tag =
      DimensionTag; ///< Dimension tag of the element
  constexpr static specfem::element::medium_tag medium_tag =
      MediumTag;                              ///< Medium tag of the element
  constexpr static bool using_simd = UseSIMD; ///< Whether to use SIMD
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
   * Here in code the actual assignment is done as follows:
   * @code {.cpp}
   * // The stress integrand is defined as follows
   * F(i, k) = T(i, 0) * dxi_k_dx + T(i, 1) * dxi_k_dz
   * // So for ncomponent = 2, we have:
   * F(0, 0) =  sigma_xx * xix + sigma_xz * xiz       // = F_{x,\xi}
   * F(0, 1) =  sigma_xz * gammax + sigma_zz * gammaz // = F_{x,\gamma}
   * F(1, 0) =  sigma_zx * xix + sigma_zz * xiz       // = F_{z,\xi}
   * F(1, 1) =  sigma_xx * gammax + sigma_xz * gammaz // = F_{x,\gamma}
   * @endcode
   *
   *
   * @param jacobian_matrix Spatial derivatives
   * @return ViewType Result of the product
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(
      const specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                            true, UseSIMD> &jacobian_matrix)
      const {
    value_type F;

    // The correct expression for F does not include Jacobian factor here.
    // However, for non regular meshes, the expression A5 in (Komatitsch et. al.
    // 2005) results in numerical instabilities. This is because spatial
    // derivatives can be small leading to precision errors. Multiplying by the
    // Jacobian factor helps normalize the result. We then avoid the jacobian
    // factor when computing the divergence in equation (A6).
    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(icomponent, 0) =
          jacobian_matrix.jacobian * (T(icomponent, 0) * jacobian_matrix.xix +
                                      T(icomponent, 1) * jacobian_matrix.xiz);
      F(icomponent, 1) = jacobian_matrix.jacobian *
                         (T(icomponent, 0) * jacobian_matrix.gammax +
                          T(icomponent, 1) * jacobian_matrix.gammaz);
    }

    return F;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const stress &other) const { return T == other.T; };

  std::string print() const {
    std::ostringstream oss;
    oss << "Stress Tensor:\n";
    for (int i = 0; i < components; ++i) {
      oss << "T(" << i << ", 0) = " << T(i, 0) << ", "
          << "T(" << i << ", 1) = " << T(i, 1) << "\n";
    }
    return oss.str();
  }
};
} // namespace point
} // namespace specfem
