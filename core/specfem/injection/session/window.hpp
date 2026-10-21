#pragma once

#include "specfem/setup.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace specfem {
namespace injection {

/**
 * @brief First step of the aligned window containing a step.
 *
 * @param istep Simulation step
 * @param window_steps Steps per window (at least 1)
 * @return The largest multiple of @p window_steps not greater than @p istep
 */
[[nodiscard]] constexpr int window_start(const int istep,
                                         const int window_steps) {
  return (istep / window_steps) * window_steps;
}

/**
 * @brief Largest window that fits a memory budget.
 *
 * The staging and device buffers each hold one window of
 * @p nrows x @p nslots values per step.
 *
 * @param memory_budget Bytes available for both buffers
 * @param nrows Injection points
 * @param nslots Slots per point
 * @param nstep Total number of simulation steps
 * @return Steps per window, in [1, nstep]; nstep when there is nothing to
 * store
 */
[[nodiscard]] constexpr int
window_steps_for_budget(const std::size_t memory_budget,
                        const std::int64_t nrows, const int nslots,
                        const int nstep) {
  const int max_steps = std::max(nstep, 1);
  if (nrows <= 0 || nslots <= 0)
    return max_steps;
  const std::size_t bytes_per_step = 2 * static_cast<std::size_t>(nrows) *
                                     static_cast<std::size_t>(nslots) *
                                     sizeof(type_real);
  return static_cast<int>(std::clamp<std::size_t>(
      memory_budget / bytes_per_step, 1, static_cast<std::size_t>(max_steps)));
}

} // namespace injection
} // namespace specfem
