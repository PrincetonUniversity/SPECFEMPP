#pragma once

#include "specfem/logger.hpp"
#include "specfem/periodic_tasks/wavefield_checkpoint.hpp"
#include "specfem/periodic_tasks/wavefield_reader.hpp"
#include "specfem/solver/forward_displacement_buffer.hpp"
#include "specfem/solver/impl/attenuation_snapshot.hpp"
#include "specfem/solver/impl/checkpointed_replay.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/tags.hpp"
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// time_marching<combined_undoatt>::run()
// ---------------------------------------------------------------------------
//
// The adjoint pass of an attenuating simulation. Because dissipative physics
// cannot be integrated backwards, the forward wavefield is not reversed here --
// it is reconstructed, window by window, from checkpoints a prior forward run
// wrote to disk. specfem::solver::impl::CheckpointedReplay does that work and
// explains how; this function sets it up and hands it one window at a time,
// last window first.
//
// PRECONDITION: a prior forward simulation wrote checkpoints of the wavefield
// and the attenuation memory variables every NT_DUMP_ATTENUATION steps.
// ---------------------------------------------------------------------------
template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined_undoatt,
                                    DimensionTag, NGLL>::run() {

  constexpr auto forward_ft = specfem::simulation::field_type::forward;
  constexpr auto adjoint_ft = specfem::simulation::field_type::adjoint;

  if (!checkpoint_reader) {
    throw std::runtime_error(
        "combined_undoatt solver: no checkpoint reader configured. "
        "Add a wavefield reader that points to forward checkpoints.");
  }

  const int checkpoint_interval = checkpoint_reader->get_time_interval();
  if (checkpoint_interval <= 0) {
    throw std::runtime_error(
        "combined_undoatt solver: wavefield reader time-interval must be > 0 "
        "(this is NT_DUMP_ATTENUATION). Set it in the wavefield config.");
  }

  // Disk checkpoints delimit the replay windows. Each window can be subdivided
  // into smaller displacement buffers, with forward state retained in memory at
  // the internal boundaries.
  const specfem::periodic_tasks::wavefield_checkpoint<DimensionTag> schedule(
      checkpoint_interval, checkpoint_buffer_subdivisions);

  const int nstep = time_scheme->get_max_timestep();
  const type_real dt = time_scheme->get_timestep();

  // ------------------------------------------------------------------
  // Mass matrices, for both the forward and the adjoint wavefield. The forward
  // wavefield carries one medium the adjoint one does not, so the two sets
  // differ.
  // ------------------------------------------------------------------
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, forward_ft>>(
            assembly, mpi_buffers, dt);
      });

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, adjoint_ft>>(
            assembly, mpi_buffers, dt);
      });

  mpi_buffers
      .template reset<specfem::data_access::DataClassType::mass_matrix>();

  // One buffer, sized for the longest leaf any window can produce, reused by
  // every window.
  specfem::solver::ForwardDisplacementBuffer<DimensionTag> displacement_buffer;
  displacement_buffer.allocate(schedule.buffer_steps(),
                               assembly.fields.forward);

  specfem::Logger::info(" -- UNDO_ATTENUATION: adjoint pass -- ");

  // The adjoint wavefield starts quiescent. Each forward replay will borrow
  // this same container and hand it back; see CheckpointedReplay.
  specfem::solver::impl::reset_attenuation_state(assembly.attenuation);

  for (const auto &task : tasks) {
    if (task) {
      task->initialize(assembly);
    }
  }

  std::ostringstream strategy_message;
  strategy_message << "Checkpoint replay: " << schedule.buffer_subdivisions()
                   << " buffer subdivisions, "
                   << schedule.checkpoint_slots(checkpoint_interval)
                   << " retained snapshots, " << schedule.buffer_steps()
                   << " buffered displacement steps";
  specfem::Logger::info(strategy_message.str());

  specfem::solver::impl::CheckpointedReplay<DimensionTag, NGLL> replay(
      assembly, mpi_buffers, *time_scheme, tasks, displacement_buffer,
      *checkpoint_reader, schedule, nstep, dt);

  // Walk the windows backwards in time, last checkpoint first. The adjoint
  // wavefield flows from one window into the next.
  const int num_windows =
      (nstep + checkpoint_interval - 1) / checkpoint_interval;

  for (int window_index = num_windows - 1; window_index >= 0; --window_index) {
    const auto [begin_step, end_step] =
        schedule.replay_window(window_index * checkpoint_interval, nstep);

    std::ostringstream message;
    message << "Running checkpoint replay for window " << window_index + 1
            << " / " << num_windows << " ("
            << schedule.forward_steps(end_step - begin_step)
            << " forward steps)";
    specfem::Logger::info(message.str());

    replay.replay_window({ begin_step, end_step });
  }

  for (const auto &task : tasks) {
    if (task) {
      task->finalize(assembly);
    }
  }

  specfem::Logger::info(" -- Simulation complete (UNDO_ATTENUATION). -- \n");

  return;
}
