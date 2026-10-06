#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk {

/**
 * @brief Cartesian component index for FK displacement and traction arrays.
 */
enum class fk_component {
  x = 0, ///< x-component
  y = 1, ///< y-component
  z = 2  ///< z-component
};

/**
 * @brief Cubic-B-spline coefficient arrays produced by the FK solver.
 *
 * This is the hand-off type consumed by sub-plan 1.2 (the Stacey injection
 * assembly container).  Sub-plan 1.2 reads the device Views directly — no
 * intermediate host deep_copy is required at injection time.
 *
 * Layout of the displacement and traction Views:
 *   @c (point_index, component, coefficient_index)
 *
 * where @c component follows @c fk_component (0 = x, 1 = y, 2 = z).
 * The pressure View is @c (point_index, coefficient_index).
 *
 * All Views live on the default Kokkos memory space.  Kernel 2 writes into
 * them on-device; for unit tests create a host mirror via
 * @c Kokkos::create_mirror_view + @c Kokkos::deep_copy.
 */
class FkResult {
public:
  /** @brief Default-construct an empty, unallocated result. */
  FkResult() = default;

  /**
   * @brief Allocate coefficient storage.
   *
   * @param number_of_points  Number of evaluation points.
   * @param coefficient_count Number of cubic-B-spline coefficients per
   * component per point.
   * @param has_traction      If true, allocate the traction coefficient View.
   * @param has_pressure      If true, allocate the scalar pressure coefficient
   * View.
   */
  FkResult(int number_of_points, int coefficient_count, bool has_traction,
           bool has_pressure);

  /**
   * @brief Return the number of evaluation points.
   * @return number of evaluation points
   */
  int number_of_points() const;

  /**
   * @brief Return the number of cubic-B-spline coefficients per component.
   * @return coefficient count
   */
  int coefficient_count() const;

  /**
   * @brief Return the resampling rate \f$N_{P,\text{resamp}}\f$.
   *
   * Set via @c set_sampling(); 0 until the driver calls it.
   *
   * @return resampling rate
   */
  int resampling_rate() const;

  /**
   * @brief Return the resampled time step \f$\Delta t_\text{min}\f$ in s.
   *
   * Set via @c set_sampling(); 0 until the driver calls it.
   *
   * @return resampled time step in s
   */
  type_real resampled_dt() const;

  /**
   * @brief Return the reference time used for t0 alignment in s.
   *
   * Set via @c set_sampling(); 0 until the driver calls it.
   *
   * @return reference time in s
   */
  type_real reference_time() const;

  /**
   * @brief Return whether traction coefficients were allocated.
   * @return true if traction View has nonzero extent
   */
  bool has_traction() const;

  /**
   * @brief Return whether pressure coefficients were allocated.
   * @return true if pressure View has nonzero extent
   */
  bool has_pressure() const;

  /**
   * @brief Displacement cubic-B-spline coefficients (const access).
   *
   * Layout: @c (point_index, component, coefficient_index).
   *
   * @return const reference to the displacement View
   */
  const Kokkos::View<type_real ***> &displacement() const;

  /**
   * @brief Displacement cubic-B-spline coefficients (mutable access for Kernel
   * 2).
   *
   * Layout: @c (point_index, component, coefficient_index).
   *
   * @return non-const reference to the displacement View
   */
  Kokkos::View<type_real ***> &displacement();

  /**
   * @brief Traction cubic-B-spline coefficients (const access).
   *
   * Layout: @c (point_index, component, coefficient_index).
   * The View has zero extent when @c has_traction() is false.
   *
   * @return const reference to the traction View
   */
  const Kokkos::View<type_real ***> &traction() const;

  /**
   * @brief Traction cubic-B-spline coefficients (mutable access for Kernel 2).
   *
   * Layout: @c (point_index, component, coefficient_index).
   * The View has zero extent when @c has_traction() is false.
   *
   * @return non-const reference to the traction View
   */
  Kokkos::View<type_real ***> &traction();

  /**
   * @brief Pressure cubic-B-spline coefficients (const access).
   *
   * Layout: @c (point_index, coefficient_index).
   * The View has zero extent when @c has_pressure() is false.
   *
   * @return const reference to the pressure View
   */
  const Kokkos::View<type_real **> &pressure() const;

  /**
   * @brief Pressure cubic-B-spline coefficients (mutable access for Kernel 2).
   *
   * Layout: @c (point_index, coefficient_index).
   * The View has zero extent when @c has_pressure() is false.
   *
   * @return non-const reference to the pressure View
   */
  Kokkos::View<type_real **> &pressure();

  /**
   * @brief Per-point time delays used for t0 alignment (const access).
   *
   * Layout: @c (point_index).
   *
   * @return const reference to the time-delay View
   */
  const Kokkos::View<type_real *> &time_delays() const;

  /**
   * @brief Per-point time delays used for t0 alignment (mutable access for
   * Kernel 2).
   *
   * Layout: @c (point_index).
   *
   * @return non-const reference to the time-delay View
   */
  Kokkos::View<type_real *> &time_delays();

  /**
   * @brief Store the sampling metadata written by the driver after allocation.
   *
   * @param resampling_rate Ratio of resampled to simulation time step.
   * @param resampled_dt    Resampled time step in s.
   * @param reference_time  Reference t0 for time alignment in s.
   */
  void set_sampling(int resampling_rate, type_real resampled_dt,
                    type_real reference_time);

private:
  Kokkos::View<type_real ***> displacement_{}; ///< (npts, 3, ncoef)
                                               ///< displacement B-spline
                                               ///< coefficients
  Kokkos::View<type_real ***> traction_{};     ///< (npts, 3, ncoef) traction
                                           ///< B-spline coefficients; empty if
                                           ///< not requested
  Kokkos::View<type_real **> pressure_{};   ///< (npts, ncoef) pressure B-spline
                                            ///< coefficients; empty if not
                                            ///< requested
  Kokkos::View<type_real *> time_delays_{}; ///< (npts) per-point t0 time delays
                                            ///< in s

  int number_of_points_ = 0;  ///< Number of evaluation points
  int coefficient_count_ = 0; ///< Number of B-spline coefficients per component
  int resampling_rate_ = 0;   ///< Resampling rate set by the driver
  type_real resampled_dt_ = 0;   ///< Resampled time step in s
  type_real reference_time_ = 0; ///< Reference time for t0 alignment in s
  bool has_traction_ = false;    ///< True if traction was allocated
  bool has_pressure_ = false;    ///< True if pressure was allocated
};

} // namespace fk
} // namespace injection
} // namespace specfem
