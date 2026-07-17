#pragma once

#include "specfem/compute.tpp"
#include "specfem/logger.hpp"
#include "specfem/periodic_tasks/wavefield_reader.hpp"
#include "specfem/solver/checkpointing_strategy.hpp"
#include "specfem/solver/forward_displacement_buffer.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/impl/update_step.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <utility>

namespace specfem::solver::combined_undoatt_impl {

template <typename ViewType> ViewType clone_view(const ViewType &view) {
  ViewType clone(std::string(view.label()) + "_undo_att_snapshot",
                 view.get_mapping());
  specfem::datatype::deep_copy(clone, view);
  return clone;
}

template <typename MediumType> struct attenuation_memory_snapshot {
  using memory_view_type = std::remove_cvref_t<
      decltype(std::declval<MediumType>().memory_variable_kappa)>;
  using strain_view_type =
      std::remove_cvref_t<decltype(std::declval<MediumType>().epsilon_xx_att)>;

  memory_view_type Rkappa;
  memory_view_type Rxx;
  memory_view_type Rxz;
  memory_view_type Ryy;
  memory_view_type Rzz;
  memory_view_type Rxy;
  memory_view_type Ryz;
  strain_view_type epsilon_xx;
  strain_view_type epsilon_zz;
  strain_view_type epsilon_xz;
  strain_view_type epsilon_yy;
  strain_view_type epsilon_xy;
  strain_view_type epsilon_yz;
};

template <typename TagsType>
using attenuation_memory_snapshot_for_tags =
    attenuation_memory_snapshot<specfem::assembly::impl::attenuation_medium<
        TagsType::dimension_tag, TagsType::medium_tag, TagsType::property_tag,
        TagsType::attenuation_tag>>;

template <typename MediumType>
auto save_attenuation_memory(const MediumType &medium) {
  attenuation_memory_snapshot<MediumType> snapshot{
    clone_view(medium.memory_variable_kappa),
    clone_view(medium.memory_variable_Rxx),
    clone_view(medium.memory_variable_Rxz),
    {},
    {},
    {},
    {},
    clone_view(medium.epsilon_xx_att),
    clone_view(medium.epsilon_zz_att),
    clone_view(medium.epsilon_xz_att),
    {},
    {},
    {}
  };

  if constexpr (requires { medium.memory_variable_Ryy; }) {
    snapshot.Ryy = clone_view(medium.memory_variable_Ryy);
    snapshot.Rzz = clone_view(medium.memory_variable_Rzz);
    snapshot.Rxy = clone_view(medium.memory_variable_Rxy);
    snapshot.Ryz = clone_view(medium.memory_variable_Ryz);
    snapshot.epsilon_yy = clone_view(medium.epsilon_yy_att);
    snapshot.epsilon_xy = clone_view(medium.epsilon_xy_att);
    snapshot.epsilon_yz = clone_view(medium.epsilon_yz_att);
  }

  return snapshot;
}

template <typename MediumType>
void restore_attenuation_memory(
    MediumType &medium,
    const attenuation_memory_snapshot<MediumType> &snapshot) {
  specfem::datatype::deep_copy(medium.memory_variable_kappa, snapshot.Rkappa);
  specfem::datatype::deep_copy(medium.memory_variable_Rxx, snapshot.Rxx);
  specfem::datatype::deep_copy(medium.memory_variable_Rxz, snapshot.Rxz);
  specfem::datatype::deep_copy(medium.epsilon_xx_att, snapshot.epsilon_xx);
  specfem::datatype::deep_copy(medium.epsilon_zz_att, snapshot.epsilon_zz);
  specfem::datatype::deep_copy(medium.epsilon_xz_att, snapshot.epsilon_xz);

  if constexpr (requires { medium.memory_variable_Ryy; }) {
    specfem::datatype::deep_copy(medium.memory_variable_Ryy, snapshot.Ryy);
    specfem::datatype::deep_copy(medium.memory_variable_Rzz, snapshot.Rzz);
    specfem::datatype::deep_copy(medium.memory_variable_Rxy, snapshot.Rxy);
    specfem::datatype::deep_copy(medium.memory_variable_Ryz, snapshot.Ryz);
    specfem::datatype::deep_copy(medium.epsilon_yy_att, snapshot.epsilon_yy);
    specfem::datatype::deep_copy(medium.epsilon_xy_att, snapshot.epsilon_xy);
    specfem::datatype::deep_copy(medium.epsilon_yz_att, snapshot.epsilon_yz);
  }

  medium.copy_to_host();
}

template <typename MediumType>
void reset_attenuation_memory(MediumType &medium) {
  Kokkos::deep_copy(medium.memory_variable_kappa, static_cast<type_real>(0));
  Kokkos::deep_copy(medium.memory_variable_Rxx, static_cast<type_real>(0));
  Kokkos::deep_copy(medium.memory_variable_Rxz, static_cast<type_real>(0));
  Kokkos::deep_copy(medium.epsilon_xx_att, static_cast<type_real>(0));
  Kokkos::deep_copy(medium.epsilon_zz_att, static_cast<type_real>(0));
  Kokkos::deep_copy(medium.epsilon_xz_att, static_cast<type_real>(0));

  if constexpr (requires { medium.memory_variable_Ryy; }) {
    Kokkos::deep_copy(medium.memory_variable_Ryy, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.memory_variable_Rzz, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.memory_variable_Rxy, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.memory_variable_Ryz, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.epsilon_yy_att, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.epsilon_xy_att, static_cast<type_real>(0));
    Kokkos::deep_copy(medium.epsilon_yz_att, static_cast<type_real>(0));
  }

  medium.copy_to_host();
}

} // namespace specfem::solver::combined_undoatt_impl

// ---------------------------------------------------------------------------
// time_marching<combined_undoatt>::run()
// ---------------------------------------------------------------------------
// Algorithm mirrors Fortran iterate_time_undoatt.F90 exactly:
//
//   PRECONDITION:
//     A prior forward simulation has written full snapshots of wavefield +
//     attenuation memory variables at checkpoint steps.
//
//   ADJOINT PASS (subset-by-subset backward):
//     For each subset window (from the last to the first):
//       (A) Save the current adjoint attenuation memory variables, then load
//           the checkpoint into fields.buffer + attenuation so forward replay
//           uses the loaded b_R_xx, b_R_xz, b_R_kappa.
//       (B) Replay forward physics over the subset, storing b_displ in the
//           RAM buffer at each step (matching Fortran
//           b_displ_elastic_buffer(:,:,it)).
//       (C) Sweep backward through the window: restore b_displ from the
//           RAM buffer, advance the adjoint wavefield one step, then call
//           compute_derivatives using (adjoint, forward) pairing.
//           The compute_derivatives call recomputes b_epsilondev on-the-fly
//           from grad(b_displ), matching Fortran compute_kernels_el lines
//           103-120 (UNDO_ATTENUATION branch).
//     Restore the adjoint attenuation memory variables before the adjoint
//     sweep.
// ---------------------------------------------------------------------------
template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined_undoatt,
                                    DimensionTag, NGLL>::run() {

  constexpr auto forward_ft = specfem::simulation::field_type::forward;
  constexpr auto adjoint_ft = specfem::simulation::field_type::adjoint;

  // ------------------------------------------------------------------
  // Locate the wavefield reader task needed for checkpoint I/O.
  // ------------------------------------------------------------------
  using CheckpointReader = specfem::periodic_tasks::periodic_task<DimensionTag>;

  CheckpointReader *checkpoint_reader = nullptr;

  auto is_checkpoint_task = [](const auto &task,
                               const CheckpointReader *reader) {
    return task.get() == reader;
  };

  for (const auto &task : tasks) {
    if (!checkpoint_reader && task &&
        task->get_type() == specfem::periodic_tasks::type::wavefield_reader) {
      checkpoint_reader = task.get();
    }
  }

  if (!checkpoint_reader) {
    throw std::runtime_error(
        "combined_undoatt solver: no wavefield_reader task found. "
        "Add a wavefield reader that points to forward checkpoints.");
  }

  // ------------------------------------------------------------------
  // Build the checkpointing strategy (fixed-stride by default).
  // The stride is the reader's time_interval, matching Fortran
  // NT_DUMP_ATTENUATION. Future: swap in a Revolve strategy here.
  // ------------------------------------------------------------------
  const int stride = checkpoint_reader->get_time_interval();
  if (stride <= 0) {
    throw std::runtime_error(
        "combined_undoatt solver: wavefield reader time-interval must be > 0 "
        "(this is NT_DUMP_ATTENUATION). Set it in the wavefield config.");
  }
  const specfem::solver::FixedStrideCheckpointing strategy(stride);

  // ------------------------------------------------------------------
  // Mass matrix initialization (same as forward::run()).
  // Both forward and adjoint fields need mass matrices.
  // ------------------------------------------------------------------
  const auto mass_dt = time_scheme->get_timestep();

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, forward_ft>>(
            assembly, mpi_buffers, mass_dt);
      });

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, adjoint_ft>>(
            assembly, mpi_buffers, mass_dt);
      });

  mpi_buffers
      .template reset<specfem::data_access::DataClassType::mass_matrix>();

  const int nstep = time_scheme->get_max_timestep();

  // Pre-allocate the RAM displacement buffer for the largest possible window.
  specfem::solver::ForwardDisplacementBuffer<DimensionTag> displ_buffer;
  displ_buffer.allocate(strategy.max_window_size(nstep),
                        assembly.fields.forward);

  // ================================================================
  // Adjoint pass — windowed forward replay + kernel accumulation
  // Matches Fortran: SIMULATION_TYPE == 3 branch of iterate_time_undoatt
  // ================================================================
  specfem::Logger::info(" -- UNDO_ATTENUATION: adjoint pass -- ");

  // Capture the constant timestep for use in compute_derivatives calls below.
  // time_scheme->get_timestep() == dt from iterate_forward() (same value).
  const type_real dt = time_scheme->get_timestep();

  // The adjoint pass starts from a fresh adjoint attenuation state, while each
  // forward replay window temporarily loads checkpointed forward attenuation
  // into the same container.
  specfem::tag_dispatch::for_each(
      assembly.attenuation.attenuation_medium_combinations,
      [&]<typename TagsType>() {
        auto &medium =
            assembly.attenuation.attenuation_storage.template get<TagsType>();
        if (medium.element_range.extent(0) > 0) {
          specfem::solver::combined_undoatt_impl::reset_attenuation_memory(
              medium);
        }
      });

  for (const auto &task : tasks) {
    if (task && !is_checkpoint_task(task, checkpoint_reader)) {
      task->initialize(assembly);
    }
  }

  // Iterate backward over subsets: last subset first, first subset last.
  // Each subset window covers [checkpoint_step, window_end).
  // We iterate over checkpoint steps in reverse order.
  int num_checkpoints = 0;
  for (int cs = 0; cs < nstep; cs += stride)
    num_checkpoints++;

  for (int subset_idx = num_checkpoints - 1; subset_idx >= 0; --subset_idx) {
    const int checkpoint_step = subset_idx * stride;
    const auto [win_start, win_end] =
        strategy.replay_window(checkpoint_step, nstep);
    const int window_size = win_end - win_start;

    // ---- (A) Load checkpoint from disk --------------------------------
    // Preserve the live adjoint attenuation memory variables before loading the
    // checkpoint, because the reader overwrites assembly.attenuation in-place.
    using AttenuationSnapshotType = specfem::tag_dispatch::TypedStorage<
        specfem::solver::combined_undoatt_impl::
            attenuation_memory_snapshot_for_tags,
        decltype(assembly.attenuation.attenuation_medium_combinations)>;

    AttenuationSnapshotType adjoint_attenuation_snapshot(
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          return specfem::solver::combined_undoatt_impl::
              save_attenuation_memory(medium);
        });

    // Loads into fields.buffer (kinematic) and assembly.attenuation (R_xx etc.)
    checkpoint_reader->run(assembly, checkpoint_step);

    std::ostringstream message;
    message << "Running forward simulation for subset " << subset_idx + 1
            << " / " << num_checkpoints;
    specfem::Logger::info(message.str());

    // Copy loaded buffer -> fields.forward (b_displ, b_veloc, b_accel)
    specfem::assembly::deep_copy(assembly.fields.forward,
                                 assembly.fields.buffer);

    // assembly.attenuation now holds the loaded forward checkpoint memory
    // variables, so update_medium<forward> replays the attenuating forward
    // physics from the checkpoint state.

    // ---- (B) Forward replay over the subset ---------------------------
    // Re-run the forward physics over [win_start, win_end), buffering b_displ.
    // Equivalent to Fortran phase 1 inside the subset loop.
    for (int k = 0; k < window_size; ++k) {
      const int sub_istep = win_start + k;

      specfem::solver::impl::apply_forward_step<NGLL, forward_ft, DimensionTag>(
          *time_scheme, assembly, mpi_buffers, sub_istep);

      // Store b_displ at slot k (forward order)
      displ_buffer.store(assembly.fields.forward, k);
    }

    // Restore the live adjoint attenuation state after forward replay.
    specfem::tag_dispatch::for_each(
        assembly.attenuation.attenuation_medium_combinations,
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          specfem::solver::combined_undoatt_impl::restore_attenuation_memory(
              medium, adjoint_attenuation_snapshot.template get<TagsType>());
        });

    // ---- (C) Adjoint sweep (reversed) ---------------------------------
    // Equivalent to Fortran phase 2 inside the subset loop.
    for (int k = window_size - 1; k >= 0; --k) {
      const int global_istep = win_start + k;

      // Restore b_displ at this sub-step (reversed order).
      // After this call, assembly.fields.forward.displacement = b_displ[k].
      // We then copy it into fields.backward so that compute_derivatives
      // (which hardcodes reading assembly.fields.backward) picks up the
      // replayed forward displacement. compute_material_derivatives only reads
      // backward.displacement, so velocity/acceleration are irrelevant here.
      // This matches Fortran compute_kernels_el lines 103-120
      // (UNDO_ATTENUATION_AND_OR_PML branch).
      displ_buffer.load(assembly.fields.forward, k);
      specfem::assembly::deep_copy(assembly.fields.backward,
                                   assembly.fields.forward);

      // Run non-checkpoint periodic tasks that should execute at this step.
      for (const auto &task : tasks) {
        if (task && !is_checkpoint_task(task, checkpoint_reader) &&
            task->should_run(global_istep + 1)) {
          task->run(assembly, global_istep + 1);
        }
      }

      // Advance adjoint wavefield one step forward.
      specfem::solver::impl::apply_adjoint_step<NGLL, adjoint_ft, DimensionTag>(
          *time_scheme, assembly, mpi_buffers, global_istep);

      // Accumulate Fréchet kernels.
      // assembly.fields.backward now holds b_displ (copied from the RAM
      // buffer via fields.forward). compute_derivatives computes
      // grad(b_displ) on-the-fly and correlates with the adjoint field to
      // accumulate rho_kl, mu_kl, kappa_kl.
      specfem::compute::compute_derivatives<NGLL,
                                            specfem::tags::Tags<DimensionTag>>(
          assembly, dt);

      specfem::solver::impl::log_time_marching_progress(global_istep, nstep);
    }
  }

  for (const auto &task : tasks) {
    if (task && !is_checkpoint_task(task, checkpoint_reader)) {
      task->finalize(assembly);
    }
  }

  specfem::Logger::info(" -- Simulation complete (UNDO_ATTENUATION). -- \n");

  return;
}
