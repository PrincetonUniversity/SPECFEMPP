#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace specfem {
namespace solver {

/**
 * @brief Abstract interface for checkpoint scheduling strategies.
 *
 * Decouples the combined undo-attenuation solver loop from the concrete
 * policy. The fixed-stride strategy is the initial implementation; an optimal
 * (Revolve/checkpointing) strategy can be plugged in later by implementing
 * this interface without touching the solver.
 *
 * Terminology (matching Fortran iterate_time_undoatt.F90):
 *   - "subset"   : one contiguous block of NT_DUMP_ATTENUATION time steps
 *   - "checkpoint step" : the first step of a subset (= the step whose full
 *                         wavefield+attenuation state is written to disk)
 */
class CheckpointingStrategy {
public:
  virtual ~CheckpointingStrategy() = default;

  /**
   * @brief Returns true if a checkpoint should be written at @p istep.
   *
   * @param istep   Current forward step (0-based, inclusive)
   * @param nstep   Total number of forward steps
   */
  virtual bool should_checkpoint(int istep, int nstep) const = 0;

  /**
   * @brief Returns the step index of the most recent checkpoint at or before
   *        @p current_step.
   *
   * Used during the adjoint pass to locate the disk file to load.
   *
   * @param current_step  Step within [0, nstep]
   * @param nstep         Total number of forward steps
   */
  virtual int last_checkpoint_before(int current_step, int nstep) const = 0;

  /**
   * @brief Returns the half-open window [window_start, window_end) that
   *        the forward replay should cover when loaded from a checkpoint at
   *        @p checkpoint_step.
   *
   * The replay marches forward from checkpoint_step to window_end-1,
   * storing one displacement snapshot per step in the RAM buffer.
   *
   * @param checkpoint_step  First step of the subset (checkpoint was written
   *                         at this step)
   * @param nstep            Total number of forward steps
   */
  virtual std::pair<int, int> replay_window(int checkpoint_step,
                                            int nstep) const = 0;

  /**
   * @brief Returns the number of steps in the largest possible replay window.
   *
   * Used to pre-allocate the RAM displacement buffer before the time loop.
   */
  virtual int max_window_size(int nstep) const = 0;
};

/**
 * @brief Fixed-stride checkpointing (Fortran NT_DUMP_ATTENUATION strategy).
 *
 * Divides the time loop into NSUBSET_ITERATIONS subsets of @p stride steps
 * each (the last subset may be shorter). A checkpoint is written at the
 * beginning of every subset (step 0, stride, 2*stride, ...).
 *
 * This is the simplest strategy and matches the Fortran
 * UNDO_ATTENUATION_AND_OR_PML implementation exactly. The interface is
 * designed so that an optimal (Revolve) strategy can replace this class later.
 */
class FixedStrideCheckpointing : public CheckpointingStrategy {
public:
  /**
   * @brief Construct with the given checkpoint stride.
   *
   * @param stride  Number of steps per subset (= NT_DUMP_ATTENUATION in
   *                Fortran). Must be >= 1.
   */
  explicit FixedStrideCheckpointing(int stride) : stride_(stride) {
    if (stride_ < 1)
      throw std::invalid_argument(
          "FixedStrideCheckpointing: stride must be >= 1");
  }

  /**
   * @brief A checkpoint is written at every @p stride steps (0, stride,
   *        2*stride, ...) and at step 0 (initial state).
   */
  bool should_checkpoint(int istep, int /*nstep*/) const override {
    return istep % stride_ == 0;
  }

  /**
   * @brief The last checkpoint at or before @p current_step is at
   *        floor(current_step / stride) * stride.
   */
  int last_checkpoint_before(int current_step, int /*nstep*/) const override {
    return (current_step / stride_) * stride_;
  }

  /**
   * @brief The replay window for a checkpoint at @p checkpoint_step covers
   *        [checkpoint_step, min(checkpoint_step + stride, nstep)).
   */
  std::pair<int, int> replay_window(int checkpoint_step,
                                    int nstep) const override {
    const int window_end = std::min(checkpoint_step + stride_, nstep);
    return { checkpoint_step, window_end };
  }

  /**
   * @brief The maximum window size equals the stride (all subsets are the same
   *        length except the last, which may be shorter).
   */
  int max_window_size(int /*nstep*/) const override { return stride_; }

  /// @brief Returns the configured stride (= NT_DUMP_ATTENUATION).
  int stride() const { return stride_; }

private:
  int stride_;
};

} // namespace solver
} // namespace specfem
