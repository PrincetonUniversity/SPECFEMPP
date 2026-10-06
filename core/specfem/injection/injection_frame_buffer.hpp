#pragma once

/**
 * @brief Uniform device-resident container for injected field values.
 *
 * Holds the time-series of displacement and/or traction components at all
 * injection evaluation points, stored as a Kokkos View on the default memory
 * space.  The device hot-path accessor @c load_on_device is method-agnostic:
 * callers do not need to know whether the values originated from an FK solve,
 * an AxiSEM3D coupling, or any other provider.
 */

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {

/**
 * @brief Uniform device buffer of injected field values for one time window.
 *
 * The layout is @c [point][component][step], stored in the default Kokkos
 * memory space.  Typical component count is 3 (displacement x/y/z) or 6
 * (displacement x/y/z followed by traction x/y/z).
 *
 * The stiffness kernel reads values through @c load_on_device, which is a
 * plain index into this buffer.  The provider (FK, AxiSEM3D, …) is solely
 * responsible for writing the correct values before the kernel runs.
 */
class InjectionFrameBuffer {
public:
  /** @brief Default-construct an empty, unallocated buffer. */
  InjectionFrameBuffer() = default;

  /**
   * @brief Allocate the buffer for the given dimensions.
   *
   * @param number_of_points     Number of injection evaluation points.
   * @param number_of_components Number of field components per point (e.g.
   *                             3 for displacement-only, 6 for
   *                             displacement + traction).
   * @param number_of_steps      Number of time steps in the window.
   */
  InjectionFrameBuffer(int number_of_points, int number_of_components,
                       int number_of_steps);

  /**
   * @brief Device hot-path read: value at (point, component) for time step
   * @p istep.
   *
   * @param istep     Time step index.
   * @param point     Evaluation point index.
   * @param component Component index.
   * @return Field value at the requested (point, component, step).
   */
  KOKKOS_INLINE_FUNCTION type_real load_on_device(int istep, int point,
                                                  int component) const {
    return values_(point, component, istep);
  }

  /**
   * @brief Return the number of evaluation points.
   * @return number of evaluation points
   */
  int number_of_points() const;

  /**
   * @brief Return the number of field components per point.
   * @return number of components
   */
  int number_of_components() const;

  /**
   * @brief Return the number of time steps in the buffer.
   * @return number of time steps
   */
  int number_of_steps() const;

  /**
   * @brief Const access to the underlying View (layout:
   * [point][component][step]).
   * @return const reference to the values View
   */
  const Kokkos::View<type_real ***> &values() const;

  /**
   * @brief Mutable access to the underlying View (for provider write-back).
   * @return non-const reference to the values View
   */
  Kokkos::View<type_real ***> &values();

private:
  Kokkos::View<type_real ***> values_{}; ///< [point][component][step] buffer
  int number_of_points_ = 0;             ///< Number of evaluation points
  int number_of_components_ = 0;         ///< Number of field components
  int number_of_steps_ = 0;              ///< Number of time steps
};

} // namespace injection
} // namespace specfem
