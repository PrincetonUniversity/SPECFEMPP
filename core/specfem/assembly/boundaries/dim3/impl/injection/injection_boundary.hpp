#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/injection/injection_provider.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Incident injection field evaluated at the Stacey boundary points.
 *
 * Owns the method-agnostic injection provider and the map from a Stacey
 * quadrature point @c (ispec_local, iz, iy, ix) to its row in the provider
 * frame buffer. Empty (@c provider == nullptr, @c number_of_points == 0) when
 * injection is disabled. The container carries no method-specific knowledge:
 * the provider alone decides how the field and traction are produced.
 *
 * @tparam DimensionTag Spatial dimension of the owning assembly.
 */
template <specfem::element::dimension_tag DimensionTag>
struct injection_boundary {
  constexpr static auto dimension_tag = DimensionTag;

  /**
   * @brief Device map from a Stacey quadrature point to its frame-buffer row.
   *
   * Dimensions @c [nspec_stacey, ngllz, nglly, ngllx], keyed identically to
   * @c stacey::quadrature_point_boundary_tag. Entry is @c -1 where the point is
   * not an injection point.
   */
  using PointIndexView =
      Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;

  std::shared_ptr<specfem::injection::injection_provider<DimensionTag>>
      provider{};                                   ///< Injection provider
  PointIndexView point_index{};                     ///< Device point-index map
  PointIndexView::host_mirror_type h_point_index{}; ///< Host mirror
  int number_of_points = 0;                         ///< Injection point count

  injection_boundary() = default;

  /** @brief Whether an injection provider is present. */
  bool enabled() const { return static_cast<bool>(provider); }

  /**
   * @brief Frame-buffer row for a Stacey quadrature point (device hot path).
   * @param ispec_local Local Stacey element index.
   * @param iz,iy,ix GLL indices.
   * @return row in @c provider->frame_buffer(), or @c -1 if not injected.
   */
  KOKKOS_INLINE_FUNCTION int point_of(int ispec_local, int iz, int iy,
                                      int ix) const {
    return point_index(ispec_local, iz, iy, ix);
  }
};

} // namespace specfem::assembly::boundaries_impl
