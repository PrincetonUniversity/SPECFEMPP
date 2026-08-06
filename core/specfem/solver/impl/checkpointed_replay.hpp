#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include "specfem/periodic_tasks/wavefield_checkpoint.hpp"
#include "specfem/setup.hpp"
#include "specfem/solver/forward_displacement_buffer.hpp"
#include "specfem/solver/impl/attenuation_snapshot.hpp"
#include "specfem/solver/mpi_buffers.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/timescheme.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Reconstructing the forward wavefield for an attenuating adjoint run
// ---------------------------------------------------------------------------
//
// Computing sensitivity kernels means correlating the adjoint wavefield, which
// marches backwards in time, against the forward wavefield at the same instant.
// Without attenuation the forward wavefield can simply be integrated backwards
// alongside the adjoint one. Attenuation makes that impossible: the physics is
// dissipative, so running it in reverse amplifies exponentially and leaves the
// memory variables in an acausal state.
//
// So instead of reversing the forward wavefield, we recompute it. A prior
// forward run wrote its full state to disk periodically; from any of those
// states the forward physics can be replayed to recover any later instant. The
// cost is roughly one extra forward simulation, and the memory is bounded by
// how much of the replay we choose to keep.
//
// Vocabulary used throughout this header and its callers:
//
//   checkpoint   A full state (wavefield + attenuation memory) written to disk
//                by the forward run, every NT_DUMP_ATTENUATION steps.
//   window       The step range one disk checkpoint opens, i.e.
//                [checkpoint_step, min(checkpoint_step + interval, nstep)).
//   segment      Any sub-range of a window handled by one recursive call.
//   leaf         A segment short enough to fit in the displacement buffer. It
//                is replayed forward once, buffering displacement at every
//                step, then swept backwards accumulating kernels.
//   retained
//   snapshot     A forward state kept in RAM at a segment boundary, so the
//                right half of a split need not be replayed from the window
//                start.
//
// All scheduling policy -- how long a leaf is, where a segment is cut, how many
// retained snapshots are affordable -- belongs to
// specfem::periodic_tasks::wavefield_checkpoint. This file only consumes those
// decisions.
//
// Reference: Komatitsch et al. (2016), and the Fortran implementation in
// iterate_time_undoatt.F90.
// ---------------------------------------------------------------------------

namespace specfem::solver::impl {

/**
 * @brief A half-open range of time steps, [begin_step, end_step).
 */
struct StepRange {
  int begin_step; ///< First step in the range
  int end_step;   ///< One past the last step in the range

  /// @brief Number of steps the range spans.
  int num_steps() const { return end_step - begin_step; }
};

/**
 * @brief A saved copy of the forward wavefield: displacement, velocity and
 * acceleration for every medium.
 *
 * Move-only, because each instance owns freshly allocated device views and
 * copying one would silently double the memory a replay holds.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class ForwardFieldSnapshot {
public:
  using forward_field_type = specfem::assembly::simulation_field<
      DimensionTag, specfem::simulation::field_type::forward>;

  ForwardFieldSnapshot() = default;

  ForwardFieldSnapshot(const ForwardFieldSnapshot &) = delete;
  ForwardFieldSnapshot &operator=(const ForwardFieldSnapshot &) = delete;
  ForwardFieldSnapshot(ForwardFieldSnapshot &&) = default;
  ForwardFieldSnapshot &operator=(ForwardFieldSnapshot &&) = default;

  /**
   * @brief Allocate a snapshot and fill it from the live forward field.
   *
   * @param field The forward wavefield to copy
   */
  explicit ForwardFieldSnapshot(const forward_field_type &field);

  /**
   * @brief Copy the saved wavefield back over the live one.
   *
   * @param field The forward wavefield to overwrite
   */
  void restore_into(forward_field_type &field) const;

private:
  using ViewType = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;

  struct MediumSnapshot {
    ViewType displacement;
    ViewType velocity;
    ViewType acceleration;
  };

  specfem::tag_dispatch::Storage<MediumSnapshot,
                                 decltype(forward_field_type::combinations)>
      storage_;
};

/**
 * @brief Everything a forward replay needs in order to resume from one instant:
 * the wavefield and the attenuation memory that goes with it.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
struct ForwardStateSnapshot {
  ForwardFieldSnapshot<DimensionTag> fields;  ///< Displacement, velocity, acc.
  AttenuationState<DimensionTag> attenuation; ///< Memory variables and strains

  /**
   * @brief Copy the saved forward state back over the live one.
   *
   * @param assembly Assembly whose forward field and attenuation are
   * overwritten
   */
  void restore_into(specfem::assembly::assembly<DimensionTag> &assembly) const;
};

/**
 * @brief Where a forward replay resumes from: either a disk checkpoint at the
 * head of a window, or a snapshot retained in memory at a segment boundary.
 *
 * Carrying the step alongside the state makes an invariant structural that used
 * to be implicit: a replay always resumes at exactly the step its origin
 * describes.
 *
 * Holds the snapshot by pointer and does not extend its lifetime. Every origin
 * is consumed within the scope that owns the snapshot it names.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> class ReplayOrigin {
public:
  /**
   * @brief An origin that reloads the disk checkpoint opening a window.
   *
   * @param step Step at which the checkpoint was written
   * @return The origin
   */
  static ReplayOrigin at_window_checkpoint(const int step) {
    return ReplayOrigin(step, nullptr);
  }

  /**
   * @brief An origin that resumes from a snapshot held in memory.
   *
   * @param step     Step the snapshot was captured at
   * @param snapshot The retained state; must outlive this origin
   * @return The origin
   */
  static ReplayOrigin
  at_snapshot(const int step,
              const ForwardStateSnapshot<DimensionTag> &snapshot) {
    return ReplayOrigin(step, &snapshot);
  }

  /// @brief Step the forward state describes.
  int step() const { return step_; }

  /// @brief Whether the state has to be reloaded from disk.
  bool is_window_checkpoint() const { return snapshot_ == nullptr; }

  /// @brief The retained state. Only valid when @ref is_window_checkpoint is
  /// false.
  const ForwardStateSnapshot<DimensionTag> &snapshot() const {
    return *snapshot_;
  }

private:
  ReplayOrigin(const int step,
               const ForwardStateSnapshot<DimensionTag> *snapshot)
      : step_(step), snapshot_(snapshot) {}

  int step_;
  const ForwardStateSnapshot<DimensionTag> *snapshot_;
};

/**
 * @brief Reconstructs the forward wavefield window by window and correlates it
 * against the adjoint wavefield to accumulate sensitivity kernels.
 *
 * One instance serves a whole adjoint pass; call @ref replay_window once per
 * window, walking the windows backwards in time. See the discussion at the top
 * of this header for what a window, a segment and a leaf are.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam NGLL Number of GLL points per element edge
 */
template <specfem::element::dimension_tag DimensionTag, int NGLL>
class CheckpointedReplay {
public:
  using PeriodicTaskType = specfem::periodic_tasks::periodic_task<DimensionTag>;
  using ScheduleType =
      specfem::periodic_tasks::wavefield_checkpoint<DimensionTag>;

  /**
   * @brief Wire the replay to the solver state it drives.
   *
   * Every argument is retained by reference for the lifetime of the replay.
   *
   * @param assembly Assembly holding the wavefields and attenuation state
   * @param mpi_buffers Solver-owned MPI exchange buffers
   * @param time_scheme Time integration scheme
   * @param tasks Periodic tasks to run during the adjoint sweep
   * @param displacement_buffer RAM buffer holding one leaf of forward
   * displacements
   * @param checkpoint_reader Reads forward checkpoints back from disk
   * @param schedule Owns all replay scheduling policy
   * @param nstep Total number of simulation steps, for progress reporting
   * @param dt Timestep, passed on to the kernel accumulation
   */
  CheckpointedReplay(
      specfem::assembly::assembly<DimensionTag> &assembly,
      specfem::solver::MPIBuffers<DimensionTag> &mpi_buffers,
      specfem::time_scheme::time_scheme &time_scheme,
      const std::vector<std::shared_ptr<PeriodicTaskType>> &tasks,
      const specfem::solver::ForwardDisplacementBuffer<DimensionTag>
          &displacement_buffer,
      PeriodicTaskType &checkpoint_reader, const ScheduleType &schedule,
      const int nstep, const type_real dt)
      : assembly_(assembly), mpi_buffers_(mpi_buffers),
        time_scheme_(time_scheme), tasks_(tasks),
        displacement_buffer_(displacement_buffer),
        checkpoint_reader_(checkpoint_reader), schedule_(schedule),
        nstep_(nstep), dt_(dt) {}

  /**
   * @brief Reconstruct one window and accumulate its contribution to the
   * kernels.
   *
   * Windows must be visited backwards in time: the adjoint attenuation state
   * evolves from the end of the simulation towards the start, and each call
   * picks up where the window to its right left off.
   *
   * @param window Half-open range of steps the disk checkpoint opens
   */
  void replay_window(const StepRange &window);

private:
  /**
   * @brief Reverse one segment, splitting it recursively when it is longer than
   * the displacement buffer.
   *
   * @param segment Half-open range of steps to reverse
   * @param origin Forward state to resume from
   * @param retained_snapshot_budget How many forward states this call may still
   * keep resident
   */
  void replay_segment(const StepRange &segment,
                      const ReplayOrigin<DimensionTag> &origin,
                      const int retained_snapshot_budget);

  /**
   * @brief Replay one leaf forward into the displacement buffer, then sweep it
   * backwards accumulating kernels.
   *
   * @param leaf Half-open range of steps, at most one buffer long
   * @param origin Forward state to resume from; may sit to the left of the leaf
   */
  void replay_buffered_leaf(const StepRange &leaf,
                            const ReplayOrigin<DimensionTag> &origin);

  /// @brief Put the live forward state back to @p origin.
  void restore_forward_state(const ReplayOrigin<DimensionTag> &origin);

  /// @brief Copy the live forward state into a new snapshot.
  ForwardStateSnapshot<DimensionTag> capture_forward_state() const;

  /// @brief Advance the forward wavefield across @p range without buffering.
  void advance_forward(const StepRange &range);

  /**
   * @brief Advance the adjoint wavefield one step and correlate it against the
   * reconstructed forward wavefield currently loaded.
   *
   * @param step Global step index being correlated
   */
  void accumulate_kernels_at(const int step);

  specfem::assembly::assembly<DimensionTag> &assembly_;
  specfem::solver::MPIBuffers<DimensionTag> &mpi_buffers_;
  specfem::time_scheme::time_scheme &time_scheme_;
  const std::vector<std::shared_ptr<PeriodicTaskType>> &tasks_;
  const specfem::solver::ForwardDisplacementBuffer<DimensionTag>
      &displacement_buffer_;
  PeriodicTaskType &checkpoint_reader_;
  const ScheduleType &schedule_;
  int nstep_;
  type_real dt_;

  StepRange window_{ 0, 0 }; ///< Window currently being replayed

  /// The adjoint attenuation state, parked here while a forward replay borrows
  /// the one live attenuation container. See @ref replay_buffered_leaf.
  AttenuationState<DimensionTag> adjoint_attenuation_;
};

} // namespace specfem::solver::impl

#include "specfem/solver/impl/checkpointed_replay.tpp"
