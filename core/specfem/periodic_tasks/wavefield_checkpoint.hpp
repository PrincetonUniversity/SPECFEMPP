#pragma once

#include "periodic_task.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Fixed-stride checkpoint schedule for undo attenuation.
 *
 * Checkpoints divide the time loop into fixed-size replay windows. The
 * inherited periodic-task interval is the checkpoint stride and is shared
 * with wavefield storage.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class wavefield_checkpoint : public periodic_task<DimensionTag> {
public:
  /**
   * @brief Construct a fixed-stride checkpoint task.
   *
   * @param time_interval Number of time steps in each replay window
   */
  explicit wavefield_checkpoint(const int time_interval)
      : periodic_task<DimensionTag>(time_interval, false) {
    if (time_interval < 1) {
      throw std::invalid_argument(
          "wavefield_checkpoint: time interval must be greater than zero");
    }
  }

  /**
   * @brief Get the replay window starting at a checkpoint.
   *
   * @param checkpoint_step Step at which the checkpoint was written
   * @param nstep Total number of simulation steps
   * @return Half-open replay window [checkpoint_step, window_end)
   */
  std::pair<int, int> replay_window(const int checkpoint_step,
                                    const int nstep) const {
    return { checkpoint_step,
             std::min(checkpoint_step + this->get_time_interval(), nstep) };
  }

  /**
   * @brief Get the maximum number of steps stored in a replay window.
   *
   * @return Fixed replay-window size
   */
  int max_window_size() const { return this->get_time_interval(); }
};

} // namespace periodic_tasks
} // namespace specfem
