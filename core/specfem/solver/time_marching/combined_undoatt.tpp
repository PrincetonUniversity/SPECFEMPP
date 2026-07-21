#pragma once

#include "specfem/compute.tpp"
#include "specfem/logger.hpp"
#include "specfem/periodic_tasks/wavefield_checkpoint.hpp"
#include "specfem/periodic_tasks/wavefield_reader.hpp"
#include "specfem/solver/forward_displacement_buffer.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/impl/update_step.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

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
}

template <typename MediumType>
void update_attenuation_snapshot(
    const MediumType &medium,
    attenuation_memory_snapshot<MediumType> &snapshot) {
  specfem::datatype::deep_copy(snapshot.Rkappa, medium.memory_variable_kappa);
  specfem::datatype::deep_copy(snapshot.Rxx, medium.memory_variable_Rxx);
  specfem::datatype::deep_copy(snapshot.Rxz, medium.memory_variable_Rxz);
  specfem::datatype::deep_copy(snapshot.epsilon_xx, medium.epsilon_xx_att);
  specfem::datatype::deep_copy(snapshot.epsilon_zz, medium.epsilon_zz_att);
  specfem::datatype::deep_copy(snapshot.epsilon_xz, medium.epsilon_xz_att);

  if constexpr (requires { medium.memory_variable_Ryy; }) {
    specfem::datatype::deep_copy(snapshot.Ryy, medium.memory_variable_Ryy);
    specfem::datatype::deep_copy(snapshot.Rzz, medium.memory_variable_Rzz);
    specfem::datatype::deep_copy(snapshot.Rxy, medium.memory_variable_Rxy);
    specfem::datatype::deep_copy(snapshot.Ryz, medium.memory_variable_Ryz);
    specfem::datatype::deep_copy(snapshot.epsilon_yy, medium.epsilon_yy_att);
    specfem::datatype::deep_copy(snapshot.epsilon_xy, medium.epsilon_xy_att);
    specfem::datatype::deep_copy(snapshot.epsilon_yz, medium.epsilon_yz_att);
  }
}

template <specfem::element::dimension_tag DimensionTag>
class forward_field_snapshot {
private:
  using forward_field_type = specfem::assembly::simulation_field<
      DimensionTag, specfem::simulation::field_type::forward>;
  using ViewType = Kokkos::View<type_real **, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace>;

  struct medium_snapshot {
    ViewType displacement;
    ViewType velocity;
    ViewType acceleration;
  };

public:
  explicit forward_field_snapshot(const forward_field_type &field) {
    specfem::tag_dispatch::for_each(
        forward_field_type::combinations, [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          const auto &source = field.template get_field<medium_tag>();
          if (source.nglob == 0)
            return;

          auto &destination = storage_.template get<TagsType>();
          auto clone_field = [](const ViewType &view) {
            ViewType clone(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              std::string(view.label()) +
                                                  "_revolve_checkpoint"),
                           view.extent(0), view.extent(1));
            Kokkos::deep_copy(clone, view);
            return clone;
          };
          destination.displacement = clone_field(source.get_field());
          destination.velocity = clone_field(source.get_field_dot());
          destination.acceleration = clone_field(source.get_field_dot_dot());
        });
  }

  void restore(forward_field_type &field) const {
    specfem::tag_dispatch::for_each(
        forward_field_type::combinations, [&]<typename TagsType>() {
          constexpr auto medium_tag = TagsType::medium_tag;
          auto &destination = field.template get_field<medium_tag>();
          if (destination.nglob == 0)
            return;

          const auto &source = storage_.template get<TagsType>();
          Kokkos::deep_copy(destination.get_field(), source.displacement);
          Kokkos::deep_copy(destination.get_field_dot(), source.velocity);
          Kokkos::deep_copy(destination.get_field_dot_dot(),
                            source.acceleration);
        });
  }

private:
  specfem::tag_dispatch::Storage<medium_snapshot,
                                 decltype(forward_field_type::combinations)>
      storage_;
};

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

  if (!checkpoint_reader) {
    throw std::runtime_error(
        "combined_undoatt solver: no checkpoint reader configured. "
        "Add a wavefield reader that points to forward checkpoints.");
  }

  // ------------------------------------------------------------------
  // Disk checkpoints delimit replay windows. A configured window can be
  // subdivided into smaller displacement buffers with retained boundary state.
  // ------------------------------------------------------------------
  const int checkpoint_interval = checkpoint_reader->get_time_interval();
  if (checkpoint_interval <= 0) {
    throw std::runtime_error(
        "combined_undoatt solver: wavefield reader time-interval must be > 0 "
        "(this is NT_DUMP_ATTENUATION). Set it in the wavefield config.");
  }
  const specfem::periodic_tasks::wavefield_checkpoint<DimensionTag>
      wavefield_checkpoint_task(checkpoint_interval,
                                checkpoint_buffer_subdivisions);

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
  displ_buffer.allocate(wavefield_checkpoint_task.buffer_steps(),
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
    if (task) {
      task->initialize(assembly);
    }
  }

  using AttenuationSnapshotType = specfem::tag_dispatch::TypedStorage<
      specfem::solver::combined_undoatt_impl::
          attenuation_memory_snapshot_for_tags,
      decltype(assembly.attenuation.attenuation_medium_combinations)>;

  // Iterate backward over subsets: last subset first, first subset last.
  // Each subset window covers [checkpoint_step, window_end).
  // We iterate over checkpoint steps in reverse order.
  const int num_checkpoints =
      (nstep + checkpoint_interval - 1) / checkpoint_interval;

  using FieldSnapshot =
      specfem::solver::combined_undoatt_impl::forward_field_snapshot<
          DimensionTag>;
  using ForwardState = std::pair<FieldSnapshot, AttenuationSnapshotType>;

  std::ostringstream strategy_message;
  strategy_message << "Checkpoint replay: "
                   << wavefield_checkpoint_task.buffer_subdivisions()
                   << " buffer subdivisions, "
                   << wavefield_checkpoint_task.checkpoint_slots(
                          checkpoint_interval)
                   << " in-memory checkpoints, "
                   << wavefield_checkpoint_task.buffer_steps()
                   << " buffered displacement steps";
  specfem::Logger::info(strategy_message.str());

  for (int subset_idx = num_checkpoints - 1; subset_idx >= 0; --subset_idx) {
    const int checkpoint_step = subset_idx * checkpoint_interval;
    const auto [win_start, win_end] =
        wavefield_checkpoint_task.replay_window(checkpoint_step, nstep);

    AttenuationSnapshotType adjoint_attenuation_snapshot(
        [&]<typename TagsType>() {
          auto &medium =
              assembly.attenuation.attenuation_storage.template get<TagsType>();
          return specfem::solver::combined_undoatt_impl::
              save_attenuation_memory(medium);
        });

    const int checkpoint_slots =
        wavefield_checkpoint_task.checkpoint_slots(win_end - win_start);
    std::vector<std::unique_ptr<ForwardState>> slots(checkpoint_slots);
    std::vector<int> free_slots;
    for (int slot = 0; slot < checkpoint_slots; ++slot)
      free_slots.push_back(slot);

    auto restore_adjoint_attenuation = [&]() {
      specfem::tag_dispatch::for_each(
          assembly.attenuation.attenuation_medium_combinations,
          [&]<typename TagsType>() {
            auto &medium = assembly.attenuation.attenuation_storage
                               .template get<TagsType>();
            specfem::solver::combined_undoatt_impl::restore_attenuation_memory(
                medium, adjoint_attenuation_snapshot.template get<TagsType>());
          });
    };

    auto update_adjoint_attenuation = [&]() {
      specfem::tag_dispatch::for_each(
          assembly.attenuation.attenuation_medium_combinations,
          [&]<typename TagsType>() {
            const auto &medium = assembly.attenuation.attenuation_storage
                                     .template get<TagsType>();
            specfem::solver::combined_undoatt_impl::update_attenuation_snapshot(
                medium, adjoint_attenuation_snapshot.template get<TagsType>());
          });
    };

    auto save_state = [&]() {
      FieldSnapshot field_snapshot(assembly.fields.forward);
      AttenuationSnapshotType attenuation_snapshot([&]<typename TagsType>() {
        auto &medium =
            assembly.attenuation.attenuation_storage.template get<TagsType>();
        return specfem::solver::combined_undoatt_impl::save_attenuation_memory(
            medium);
      });
      return std::make_unique<ForwardState>(std::move(field_snapshot),
                                            std::move(attenuation_snapshot));
    };

    auto restore_state = [&](int slot) {
      slots[slot]->first.restore(assembly.fields.forward);
      specfem::tag_dispatch::for_each(
          assembly.attenuation.attenuation_medium_combinations,
          [&]<typename TagsType>() {
            auto &medium = assembly.attenuation.attenuation_storage
                               .template get<TagsType>();
            specfem::solver::combined_undoatt_impl::restore_attenuation_memory(
                medium, slots[slot]->second.template get<TagsType>());
          });
    };

    auto restore_base = [&](int slot) {
      if (slot >= 0) {
        restore_state(slot);
      } else {
        checkpoint_reader->run(assembly, checkpoint_step);
        specfem::assembly::deep_copy(assembly.fields.forward,
                                     assembly.fields.buffer);
      }
    };

    auto advance = [&](int start, int count) {
      for (int offset = 0; offset < count; ++offset) {
        specfem::solver::impl::apply_forward_step<NGLL, forward_ft,
                                                  DimensionTag>(
            *time_scheme, assembly, mpi_buffers, start + offset);
      }
    };

    auto correlate_adjoint_state = [&](int global_istep) {
      specfem::assembly::deep_copy(assembly.fields.backward,
                                   assembly.fields.forward);

      // Run periodic tasks that should execute at this step.
      for (const auto &task : tasks) {
        if (task && task->should_run(global_istep + 1)) {
          task->run(assembly, global_istep + 1);
        }
      }

      specfem::solver::impl::apply_adjoint_step<NGLL, adjoint_ft, DimensionTag>(
          *time_scheme, assembly, mpi_buffers, global_istep);
      specfem::compute::compute_derivatives<NGLL,
                                            specfem::tags::Tags<DimensionTag>>(
          assembly, dt);
      specfem::solver::impl::log_time_marching_progress(global_istep, nstep);
    };

    auto reverse_buffered_leaf = [&](int base_start, int leaf_start,
                                     int leaf_end, int base_slot) {
      restore_base(base_slot);
      advance(base_start, leaf_start - base_start);
      for (int step = leaf_start; step < leaf_end; ++step) {
        advance(step, 1);
        displ_buffer.store(assembly.fields.forward, step - leaf_start);
      }

      restore_adjoint_attenuation();
      for (int step = leaf_end - 1; step >= leaf_start; --step) {
        displ_buffer.load(assembly.fields.forward, step - leaf_start);
        correlate_adjoint_state(step);
      }
      if (leaf_start != win_start)
        update_adjoint_attenuation();
    };

    std::ostringstream message;
    message << "Running checkpoint replay for subset " << subset_idx + 1
            << " / " << num_checkpoints << " ("
            << wavefield_checkpoint_task.forward_steps(win_end - win_start)
            << " forward steps)";
    specfem::Logger::info(message.str());

    if (checkpoint_slots == 0) {
      reverse_buffered_leaf(win_start, win_start, win_end, -1);
    } else {
      std::function<void(int, int, int, int)> reverse_interval;
      reverse_interval = [&](int start, int end, int available, int base_slot) {
        const int length = end - start;
        if (length <= 0)
          return;

        if (length <= wavefield_checkpoint_task.buffer_steps()) {
          reverse_buffered_leaf(start, start, end, base_slot);
          return;
        }

        if (available == 0) {
          for (int leaf_end = end; leaf_end > start;
               leaf_end -= wavefield_checkpoint_task.buffer_steps()) {
            const int leaf_start =
                std::max(start,
                         leaf_end - wavefield_checkpoint_task.buffer_steps());
            reverse_buffered_leaf(start, leaf_start, leaf_end, base_slot);
          }
          return;
        }

        const int left_length = wavefield_checkpoint_task.split(length);
        restore_base(base_slot);
        advance(start, left_length);
        const int split_slot = free_slots.back();
        free_slots.pop_back();
        slots[split_slot] = save_state();

        reverse_interval(start + left_length, end, available - 1, split_slot);
        slots[split_slot].reset();
        free_slots.push_back(split_slot);
        reverse_interval(start, start + left_length, available, base_slot);
      };
      reverse_interval(win_start, win_end, checkpoint_slots, -1);
    }
  }

  for (const auto &task : tasks) {
    if (task) {
      task->finalize(assembly);
    }
  }

  specfem::Logger::info(" -- Simulation complete (UNDO_ATTENUATION). -- \n");

  return;
}
