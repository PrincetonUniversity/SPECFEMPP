#pragma once

#include "enumerations/interface.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {

/**
 * @brief GLL quadrature points and weights for spectral elements.
 *
 * Stores Gauss-Lobatto-Legendre quadrature data using Kokkos views
 * for device/host access.
 */
template <specfem::dimension::type DimensionTag> struct GLLQuadrature {
  constexpr static auto dimension_tag = DimensionTag;

  using ViewType = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  using DViewType = Kokkos::View<type_real **, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>;

  int N;                          ///< Number of GLL points
  ViewType xi;                    ///< Device GLL points on [-1,1]
  ViewType weights;               ///< Device integration weights
  DViewType hprime;               ///< Device Lagrange derivative matrix
  DViewType::HostMirror h_hprime; ///< Host Lagrange derivative matrix
  ViewType::HostMirror h_xi;      ///< Host GLL points
  ViewType::HostMirror h_weights; ///< Host weights

  GLLQuadrature() = default;

  /**
   * @brief Constructor from quadrature object.
   *
   * Copies GLL data from source quadrature to device/host views.
   */
  GLLQuadrature(const specfem::quadrature::quadratures &quadratures)
      : N(quadratures.gll.get_N()), xi(quadratures.gll.get_xi()),
        weights(quadratures.gll.get_w()), h_xi(quadratures.gll.get_hxi()),
        h_weights(quadratures.gll.get_hw()),
        hprime(quadratures.gll.get_hprime()),
        h_hprime(quadratures.gll.get_hhprime()) {}
};

/**
 * @brief Assembly quadrature wrapper for mesh integration.
 *
 * Inherits GLL quadrature functionality for use in assembly operations.
 */
template <specfem::dimension::type DimensionTag>
struct quadrature
    : public specfem::assembly::mesh_impl::GLLQuadrature<DimensionTag> {
public:
  constexpr static auto dimension_tag = DimensionTag;

  quadrature() = default;

  /**
   * @brief Constructor from quadrature object.
   */
  quadrature(const specfem::quadrature::quadratures &quadratures)
      : specfem::assembly::mesh_impl::GLLQuadrature<DimensionTag>(quadratures) {
  }
};

} // namespace specfem::assembly::mesh_impl
