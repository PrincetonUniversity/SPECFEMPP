#pragma once

#include "specfem/compute.tpp"
#include "specfem/solver/impl/checkpointed_replay.hpp"
#include "specfem/solver/impl/update_step.hpp"
#include "specfem/tags.hpp"
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag>
specfem::solver::impl::ForwardFieldSnapshot<DimensionTag>::ForwardFieldSnapshot(
    const forward_field_type &field) {
  specfem::tag_dispatch::for_each(
      forward_field_type::combinations, [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        const auto &source = field.template get_field<medium_tag>();
        if (source.nglob == 0)
          return;

        auto &destination = storage_.template get<TagsType>();
        auto clone_field = [](const ViewType &view) {
          ViewType clone(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 std::string(view.label()) + "_snapshot"),
              view.extent(0), view.extent(1));
          Kokkos::deep_copy(clone, view);
          return clone;
        };
        destination.displacement = clone_field(source.get_field());
        destination.velocity = clone_field(source.get_field_dot());
        destination.acceleration = clone_field(source.get_field_dot_dot());
      });
}

template <specfem::element::dimension_tag DimensionTag>
void specfem::solver::impl::ForwardFieldSnapshot<DimensionTag>::restore_into(
    forward_field_type &field) const {
  specfem::tag_dispatch::for_each(
      forward_field_type::combinations, [&]<typename TagsType>() {
        constexpr auto medium_tag = TagsType::medium_tag;
        auto &destination = field.template get_field<medium_tag>();
        if (destination.nglob == 0)
          return;

        const auto &source = storage_.template get<TagsType>();
        Kokkos::deep_copy(destination.get_field(), source.displacement);
        Kokkos::deep_copy(destination.get_field_dot(), source.velocity);
        Kokkos::deep_copy(destination.get_field_dot_dot(), source.acceleration);
      });
}

template <specfem::element::dimension_tag DimensionTag>
void specfem::solver::impl::ForwardStateSnapshot<DimensionTag>::restore_into(
    specfem::assembly::assembly<DimensionTag> &assembly) const {
  fields.restore_into(assembly.fields.forward);
  attenuation.restore_into(assembly.attenuation);
}

// ---------------------------------------------------------------------------
// Driving one window
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<
    DimensionTag, NGLL>::replay_window(const StepRange &window) {

  window_ = window;

  // The adjoint wavefield has been marching backwards through the windows to
  // our right and its attenuation memory is live right now. Park it: the
  // forward replay below needs the one attenuation container for itself.
  adjoint_attenuation_ = AttenuationState<DimensionTag>{};
  adjoint_attenuation_ =
      AttenuationState<DimensionTag>::capture_from(assembly_.attenuation);

  replay_segment(
      window,
      ReplayOrigin<DimensionTag>::at_window_checkpoint(window.begin_step),
      schedule_.checkpoint_slots(window.num_steps()));
}

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<DimensionTag, NGLL>::
    replay_segment(const StepRange &segment,
                   const ReplayOrigin<DimensionTag> &origin,
                   const int retained_snapshot_budget) {

  const int num_steps = segment.num_steps();
  if (num_steps <= 0)
    return;

  // Short enough to reconstruct in a single pass through the buffer.
  if (num_steps <= schedule_.buffer_steps()) {
    replay_buffered_leaf(segment, origin);
    return;
  }

  // Nothing left to spend on retained states, so walk the leaves from right to
  // left, replaying from `origin` every time. Quadratic in forward steps but
  // needs no extra memory. The current schedule always hands out exactly enough
  // budget to avoid this, but a stingier one would still terminate correctly.
  if (retained_snapshot_budget == 0) {
    for (int leaf_end = segment.end_step; leaf_end > segment.begin_step;
         leaf_end -= schedule_.buffer_steps()) {
      const int leaf_begin =
          std::max(segment.begin_step, leaf_end - schedule_.buffer_steps());
      replay_buffered_leaf({ leaf_begin, leaf_end }, origin);
    }
    return;
  }

  // Cut the segment in two and reverse the right half first, so that when the
  // left half runs it can still resume from the origin the whole segment used.
  const int split_step = segment.begin_step + schedule_.split(num_steps);

  restore_forward_state(origin);
  advance_forward({ origin.step(), split_step });

  {
    const ForwardStateSnapshot<DimensionTag> retained = capture_forward_state();
    replay_segment(
        { split_step, segment.end_step },
        ReplayOrigin<DimensionTag>::at_snapshot(split_step, retained),
        retained_snapshot_budget - 1);
  } // retained state released here, before the left half runs, so the number
    // resident at once never exceeds the recursion depth

  replay_segment({ segment.begin_step, split_step }, origin,
                 retained_snapshot_budget);
}

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<DimensionTag, NGLL>::
    replay_buffered_leaf(const StepRange &leaf,
                         const ReplayOrigin<DimensionTag> &origin) {

  // Resume the forward wavefield and run it up to the leaf. Nothing in this
  // stretch is buffered -- it is only here to reach the leaf's first step.
  restore_forward_state(origin);
  advance_forward({ origin.step(), leaf.begin_step });

  // Replay the leaf itself, keeping every displacement. Displacement alone is
  // enough: the kernels need only it and the strain, and the strain is
  // recomputed from its gradient during the backward sweep.
  for (int step = leaf.begin_step; step < leaf.end_step; ++step) {
    advance_forward({ step, step + 1 });
    displacement_buffer_.store(assembly_.fields.forward,
                               step - leaf.begin_step);
  }

  // The forward replay is finished with the attenuation container, so hand it
  // back to the adjoint wavefield before sweeping backwards through the leaf.
  adjoint_attenuation_.restore_into(assembly_.attenuation);
  for (int step = leaf.end_step - 1; step >= leaf.begin_step; --step) {
    displacement_buffer_.load(assembly_.fields.forward, step - leaf.begin_step);
    accumulate_kernels_at(step);
  }

  // Park the evolved adjoint attenuation state again for the leaf to our left.
  // The leaf that opens the window has none, and the next window captures the
  // live container directly.
  if (leaf.begin_step != window_.begin_step) {
    adjoint_attenuation_.refresh_from(assembly_.attenuation);
  }
}

// ---------------------------------------------------------------------------
// The forward wavefield
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<DimensionTag, NGLL>::
    restore_forward_state(const ReplayOrigin<DimensionTag> &origin) {

  if (!origin.is_window_checkpoint()) {
    origin.snapshot().restore_into(assembly_);
    return;
  }

  // The reader lands the wavefield in the buffer field and the attenuation
  // memory directly in its own container, so only the wavefield half needs
  // moving into place.
  checkpoint_reader_.run(assembly_, origin.step());
  specfem::assembly::deep_copy(assembly_.fields.forward,
                               assembly_.fields.buffer);
}

template <specfem::element::dimension_tag DimensionTag, int NGLL>
specfem::solver::impl::ForwardStateSnapshot<DimensionTag>
specfem::solver::impl::CheckpointedReplay<DimensionTag,
                                          NGLL>::capture_forward_state() const {
  return ForwardStateSnapshot<DimensionTag>{
    ForwardFieldSnapshot<DimensionTag>(assembly_.fields.forward),
    AttenuationState<DimensionTag>::capture_from(assembly_.attenuation)
  };
}

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<
    DimensionTag, NGLL>::advance_forward(const StepRange &range) {
  for (int step = range.begin_step; step < range.end_step; ++step) {
    specfem::solver::impl::apply_forward_step<
        NGLL, specfem::simulation::field_type::forward, DimensionTag>(
        time_scheme_, assembly_, mpi_buffers_, step);
  }
}

// ---------------------------------------------------------------------------
// The adjoint wavefield
// ---------------------------------------------------------------------------

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::impl::CheckpointedReplay<
    DimensionTag, NGLL>::accumulate_kernels_at(const int step) {

  // Kernel evaluation reads the reconstructed forward wavefield through the
  // backward field, which is where the non-attenuating solver leaves it.
  specfem::assembly::deep_copy(assembly_.fields.backward,
                               assembly_.fields.forward);

  for (const auto &task : tasks_) {
    if (task && task->should_run(step + 1)) {
      task->run(assembly_, step + 1);
    }
  }

  specfem::solver::impl::apply_adjoint_step<
      NGLL, specfem::simulation::field_type::adjoint, DimensionTag>(
      time_scheme_, assembly_, mpi_buffers_, step);
  specfem::compute::compute_derivatives<NGLL,
                                        specfem::tags::Tags<DimensionTag>>(
      assembly_, dt_);
  specfem::solver::impl::log_time_marching_progress(step, nstep_);
}
