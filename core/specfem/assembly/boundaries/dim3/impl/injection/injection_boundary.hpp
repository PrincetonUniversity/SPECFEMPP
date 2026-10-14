#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Device-resident incident injection field at the Stacey boundary
 * points.
 *
 * Holds only device-capturable state: the map from a Stacey quadrature point
 * @c (ispec_local, iz, iy, ix) to its row in the provider frame buffer, a copy
 * of the compute→local Stacey element map, and the current @c
 * InjectionFrameBuffer window. This struct is captured by value into the
 * stiffness device kernel, so it must stay free of host-only members (the
 * owning provider @c shared_ptr lives on the assembly, not here). Empty (@c
 * number_of_points == 0) when injection is disabled. Carries no method-specific
 * knowledge: the provider alone decides how the field and traction are
 * produced.
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

  /** @brief Compute→local Stacey element map (shallow copy of the boundaries').
   */
  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

  PointIndexView point_index{};                     ///< Device point-index map
  PointIndexView::host_mirror_type h_point_index{}; ///< Host mirror
  IndexView stacey_index_mapping{}; ///< Compute→local Stacey map
  specfem::injection::InjectionFrameBuffer frame{}; ///< Current incident-field
                                                    ///< window (device;
                                                    ///< refreshed per step by
                                                    ///< the solver)
  int number_of_points = 0;                         ///< Injection point count

  injection_boundary() = default;

  /** @brief Whether an injection field is present at the Stacey points. */
  KOKKOS_INLINE_FUNCTION bool enabled() const { return number_of_points > 0; }

  /**
   * @brief Frame-buffer row for a Stacey quadrature point (device hot path).
   * @param ispec_local Local Stacey element index.
   * @param iz,iy,ix GLL indices.
   * @return row in @c frame, or @c -1 if not injected.
   */
  KOKKOS_INLINE_FUNCTION int point_of(int ispec_local, int iz, int iy,
                                      int ix) const {
    return point_index(ispec_local, iz, iy, ix);
  }
};

} // namespace specfem::assembly::boundaries_impl
