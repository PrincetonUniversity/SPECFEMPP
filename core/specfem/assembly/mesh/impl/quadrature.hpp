#pragma once

#include "enumerations/interface.hpp"
#include "quadrature/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief Gauss-Lobatto-Legendre quadrature points and weights
 *
 * @tparam DimensionTag Dimension tag (2D or 3D)
 */
template <specfem::dimension::type DimensionTag> struct GLLQuadrature {
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag

  using ViewType = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  using DViewType = Kokkos::View<type_real **, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>;

  int N;                          ///< Number of quadrature points
  ViewType xi;                    ///< Quadrature points
  ViewType weights;               ///< Quadrature weights
  DViewType hprime;               ///< Derivative of lagrange interpolants
  DViewType::HostMirror h_hprime; ///< Derivative of lagrange interpolants
  ViewType::HostMirror h_xi;      ///< Quadrature points
  ViewType::HostMirror h_weights; ///< Quadrature weights

  /**
   * @brief Construct a new GLLQuadrature object
   *
   */
  GLLQuadrature() = default;

  /**
   * @brief Construct a new GLLQuadrature object
   *
   * @param quadratures quadratures object
   */
  GLLQuadrature(const specfem::quadrature::quadratures &quadratures)
      : N(quadratures.gll.get_N()), xi(quadratures.gll.get_xi()),
        weights(quadratures.gll.get_w()), h_xi(quadratures.gll.get_hxi()),
        h_weights(quadratures.gll.get_hw()),
        hprime(quadratures.gll.get_hprime()),
        h_hprime(quadratures.gll.get_hhprime()) {}
};

/**
 * @brief Information about the integration quadratures
 *
 * This struct provides access to the quadrature points and weights for a given
 * dimension.
 */
template <specfem::dimension::type DimensionTag>
struct quadrature
    : public specfem::assembly::mesh_impl::GLLQuadrature<DimensionTag> {
public:
  constexpr static auto dimension_tag = DimensionTag; ///< Dimension tag

  quadrature() = default;

  /**
   * @brief Construct a new quadrature object
   *
   * @param quadratures quadratures object
   */
  quadrature(const specfem::quadrature::quadratures &quadratures)
      : specfem::assembly::mesh_impl::GLLQuadrature<DimensionTag>(quadratures) {
  }
};

} // namespace specfem::assembly::mesh_impl
