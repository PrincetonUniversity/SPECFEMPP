#pragma once

#include "periodic_task.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Checkpoint replay schedule for undo attenuation.
 *
 * Checkpoints divide the time loop into fixed-size replay windows. The
 * inherited periodic-task interval is the checkpoint stride and is shared
 * with wavefield storage. Each window may be subdivided into bounded
 * displacement buffers with full forward-state checkpoints retained at the
 * internal boundaries. One subdivision performs direct full-window replay.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class wavefield_checkpoint : public periodic_task<DimensionTag> {
public:
  /**
   * @brief Construct a checkpoint replay task.
   *
   * @param time_interval Number of time steps in each replay window
   * @param buffer_subdivisions Requested number of buffer subdivisions
   */
  explicit wavefield_checkpoint(const int time_interval,
                                const int buffer_subdivisions = 1)
      : periodic_task<DimensionTag>(validate_time_interval(time_interval),
                                    false),
        buffer_subdivisions_(std::min(
            validate_subdivisions(buffer_subdivisions), time_interval)),
        buffer_steps_((time_interval + buffer_subdivisions_ - 1) /
                      buffer_subdivisions_) {}

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
   * @brief Get the replay-window size before subdivision.
   *
   * @return Persistent checkpoint interval
   */
  int max_window_size() const { return this->get_time_interval(); }

  /// @brief Get the effective number of subdivisions per replay window.
  int buffer_subdivisions() const { return buffer_subdivisions_; }

  /// @brief Get the maximum number of displacements in a replay leaf.
  int buffer_steps() const { return buffer_steps_; }

  /**
   * @brief Get the internal checkpoint slots needed by an interval.
   *
   * @param num_steps Length of the replay interval
   * @return Number of resident full-state checkpoint slots
   */
  int checkpoint_slots(const int num_steps) const {
    validate_num_steps(num_steps);
    if (num_steps == 0)
      return 0;
    return (num_steps + buffer_steps_ - 1) / buffer_steps_ - 1;
  }

  /**
   * @brief Get the first leaf length for an interval.
   *
   * The short leaf is placed first so that all later leaves have the maximum
   * buffer size and the last retained checkpoint is as early as possible.
   *
   * @param num_steps Length of the replay interval
   * @return First leaf length, or zero for a single-leaf interval
   */
  int split(const int num_steps) const {
    validate_num_steps(num_steps);
    if (num_steps <= buffer_steps_)
      return 0;
    const int remainder = num_steps % buffer_steps_;
    return remainder == 0 ? buffer_steps_ : remainder;
  }

  /**
   * @brief Get the forward advances required by replay.
   *
   * @param num_steps Length of the replay interval
   * @return Number of forward time-step advances
   */
  std::uint64_t forward_steps(const int num_steps) const {
    validate_num_steps(num_steps);
    if (num_steps <= buffer_steps_)
      return num_steps;
    return static_cast<std::uint64_t>(2) * num_steps - buffer_steps_;
  }

private:
  static int validate_time_interval(const int time_interval) {
    if (time_interval < 1) {
      throw std::invalid_argument(
          "wavefield_checkpoint: time interval must be greater than zero");
    }
    return time_interval;
  }

  static int validate_subdivisions(const int buffer_subdivisions) {
    if (buffer_subdivisions < 1) {
      throw std::invalid_argument(
          "wavefield_checkpoint: subdivisions must be positive");
    }
    return buffer_subdivisions;
  }

  void validate_num_steps(const int num_steps) const {
    if (num_steps < 0 || num_steps > this->get_time_interval()) {
      throw std::out_of_range(
          "wavefield_checkpoint: interval exceeds configured size");
    }
  }

  int buffer_subdivisions_;
  int buffer_steps_;
};

} // namespace periodic_tasks
} // namespace specfem
