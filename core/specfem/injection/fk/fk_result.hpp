#pragma once

#include "specfem/setup.hpp"
#include "specfem/utilities/cubic_bspline.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk {

/**
 * @brief Depth-grouped FK result with on-device sampling.
 *
 * The FK field factorises as @f$u(x,y,z,t) = U(z, t - \tau(x,y))@f$, so the
 * depth-only series @f$U(z,\cdot)@f$ are stored once per (medium, depth) group
 * as cubic-B-spline coefficients, and each evaluation point carries only its
 * group index, an abscissa offset @f$\text{pad} - \tau/\Delta t_\text{min}@f$,
 * the traction material factors, and a medium flag. The only device kernel in
 * the FK path is @c sample, which reconstructs @f$u(t-\tau)@f$ per point by
 * evaluating the group's splines at the shifted abscissa and applying the
 * azimuth and traction rotations.
 *
 * Built by @c specfem::injection::fk_impl::make_result; consumed by the
 * injection provider.
 */
class FkResult {
public:
  /** @brief Default-construct an empty, unallocated result. */
  FkResult() = default;

  /**
   * @brief Construct from device-resident arrays.
   *
   * @param series_coefficients (group, series, coefficient), LayoutRight.
   * @param point_group         (point) group index.
   * @param point_offset        (point) abscissa offset pad - tau/dt_min.
   * @param point_material      (point, 6) xi1, xim, bulk, nx, ny, nz.
   * @param point_is_acoustic   (point) 1 for acoustic, 0 for elastic.
   * @param cos_phi, sin_phi    Azimuth rotation (shared by all points).
   * @param resampling_rate     N_P (SEM-step to resampled-grid ratio).
   * @param has_traction        Whether traction components are meaningful.
   */
  FkResult(Kokkos::View<type_real ***, Kokkos::LayoutRight> series_coefficients,
           Kokkos::View<int *> point_group, Kokkos::View<double *> point_offset,
           Kokkos::View<type_real **, Kokkos::LayoutRight> point_material,
           Kokkos::View<int *> point_is_acoustic, type_real cos_phi,
           type_real sin_phi, int resampling_rate, bool has_traction);

  int number_of_points() const { return number_of_points_; }
  int number_of_groups() const { return number_of_groups_; }
  int coefficient_count() const { return coefficient_count_; }
  int resampling_rate() const { return resampling_rate_; }
  bool has_traction() const { return has_traction_; }
  type_real cos_phi() const { return cos_phi_; }
  type_real sin_phi() const { return sin_phi_; }

  const Kokkos::View<type_real ***, Kokkos::LayoutRight> &
  series_coefficients() const {
    return series_coefficients_;
  }
  const Kokkos::View<int *> &point_group() const { return point_group_; }
  const Kokkos::View<double *> &point_offset() const { return point_offset_; }

  /**
   * @brief Reconstruct the field for a window of SEM steps, on device.
   *
   * Fills @p out(point, component, k) for SEM steps
   * @c [first_step, first_step + number_of_steps). Components:
   * 0..2 displacement/velocity (cos_phi*S0, sin_phi*S0, S1); 3..5 traction
   * (when @c out.extent(1) >= 6); 6 pressure (when @c out.extent(1) >= 7).
   *
   * @param first_step      First SEM step in the window.
   * @param number_of_steps Window length in SEM steps.
   * @param out             Output view (point, component, step-in-window).
   */
  void sample(int first_step, int number_of_steps,
              const Kokkos::View<type_real ***> &out) const;

private:
  Kokkos::View<type_real ***, Kokkos::LayoutRight> series_coefficients_{};
  Kokkos::View<int *> point_group_{};
  Kokkos::View<double *> point_offset_{};
  Kokkos::View<type_real **, Kokkos::LayoutRight> point_material_{};
  Kokkos::View<int *> point_is_acoustic_{};
  type_real cos_phi_ = 0;
  type_real sin_phi_ = 0;
  int resampling_rate_ = 0;
  int number_of_points_ = 0;
  int number_of_groups_ = 0;
  int coefficient_count_ = 0;
  bool has_traction_ = false;
};

} // namespace fk
} // namespace injection
} // namespace specfem
