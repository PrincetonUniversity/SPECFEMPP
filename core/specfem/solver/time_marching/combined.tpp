#pragma once

#include "specfem/compute.tpp"
#include "specfem/logger.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/impl/update_step.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/tags.hpp"

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionTag, NGLL>::run() {

  constexpr auto adjoint = specfem::simulation::field_type::adjoint;
  constexpr auto backward = specfem::simulation::field_type::backward;

  // Compute and invert the mass matrix for both wavefields.
  const auto mass_dt = time_scheme->get_timestep();
  const auto mass_set =
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
      MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic) *
      WAVEFIELD_SET(adjoint, backward);

  specfem::tag_dispatch::for_each(mass_set, [&]<typename ElementTags>() {
    specfem::solver::impl::init_medium_mass<NGLL, ElementTags>(
        assembly, mpi_buffers, mass_dt);
  });

  // The mass-matrix buffers are no longer needed after assembly + inversion.
  mpi_buffers
      .template reset<specfem::data_access::DataClassType::mass_matrix>();

  const int nstep = time_scheme->get_max_timestep();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto &task : tasks) {
    if (task && !task->should_run(nstep) && task->should_run(-1)) {
      task->run(assembly, nstep);
    }
  }

  // Initialize the backward wavefield from the final forward wavefield buffer.
  // The pre-loop task above loads StepNSTEP only when it is not also on the
  // regular task cadence; otherwise the first in-loop task refresh below does.
  specfem::assembly::deep_copy(assembly.fields.backward,
                               assembly.fields.buffer);

  for (const auto [istep, dt] : time_scheme->iterate_backward()) {
    for (const auto &task : tasks) {
      if (task && task->should_run(istep + 1)) {
        task->run(assembly, istep + 1);
      }
    }

    if (istep == nstep - 1) {
      specfem::assembly::deep_copy(assembly.fields.backward,
                                   assembly.fields.buffer);
    }

    // Adjoint time step.
    specfem::solver::impl::apply_adjoint_step<NGLL, adjoint, DimensionTag>(
        *time_scheme, assembly, mpi_buffers, istep);

    // Compute kernels using adjoint at t=(j+1)*dt and backward at
    // t=(nstep-j)*dt. This matches the Fortran kernel wavefield pairing after
    // the reconstructed forward field has been loaded.
    specfem::compute::compute_derivatives<NGLL,
                                          specfem::tags::Tags<DimensionTag>>(
        assembly, dt);

    // Backward (reverse-time) reconstruction step.
    specfem::solver::impl::apply_backward_step<NGLL, backward, DimensionTag>(
        *time_scheme, assembly, mpi_buffers, istep);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      specfem::compute::compute_seismograms<
          NGLL, specfem::tags::Tags<DimensionTag, backward>>(
          assembly, time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }

    specfem::solver::impl::log_time_marching_progress(istep, nstep);
  }

  for (const auto &task : tasks) {
    task->finalize(assembly);
  }

  specfem::Logger::info(" -- Simulation complete. -- \n");

  return;
}
