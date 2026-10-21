#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {

/**
 * @brief Device-resident window of the incident field.
 *
 * Holds the steps [window_start, window_start + nsteps) for every row and
 * slot. Cheap to copy, so kernels capture it by value.
 */
struct FrameBuffer {
  /** @brief Device storage, indexed [row][slot][step in window]. */
  using view_type = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>;

  view_type data;       ///< Incident field for the current window
  int window_start = 0; ///< First simulation step held in data
  int nsteps = 0;       ///< Number of valid steps in data

  /**
   * @brief Incident field value at a simulation step.
   *
   * @param istep Simulation step; must lie in the current window
   * @param row Injection point (row of the point list)
   * @param slot Component slot (see specfem::injection::Request)
   * @return The value provided by the plugin
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION type_real load(const int istep,
                                                      const int row,
                                                      const int slot) const {
#ifndef NDEBUG
    if (istep < window_start || istep >= window_start + nsteps)
      Kokkos::abort("specfem::injection::FrameBuffer::load: step outside the "
                    "current window");
#endif
    return data(row, slot, istep - window_start);
  }
};

} // namespace injection
} // namespace specfem
