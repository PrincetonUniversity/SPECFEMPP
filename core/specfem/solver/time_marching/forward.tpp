#pragma once

#include "specfem/compute.tpp"
#include "specfem/logger.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/solver/impl/update_step.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/tags.hpp"

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::forward,
                                    DimensionTag, NGLL>::run() {

  constexpr auto forward = specfem::simulation::field_type::forward;

  // Compute and invert the mass matrix.
  const auto mass_dt = time_scheme->get_timestep();

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic,
                     elastic_psv_t, elastic_spin),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, forward>>(
            assembly, mpi_buffers, mass_dt);
      });

  // The mass-matrix buffers are no longer needed after assembly + inversion.
  mpi_buffers
      .template reset<specfem::data_access::DataClassType::mass_matrix>();

  const int nstep = time_scheme->get_max_timestep();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    specfem::solver::impl::apply_forward_step<NGLL, forward, DimensionTag>(
        *time_scheme, assembly, mpi_buffers, istep);

    // Compute seismograms if required
    if (time_scheme->compute_seismogram(istep)) {
      specfem::compute::compute_seismograms<
          NGLL, specfem::tags::Tags<DimensionTag, forward>>(
          assembly, time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }
    // Run periodic tasks such as plotting, etc.
    for (const auto &task : tasks) {
      if (task && task->should_run(istep + 1)) {
        task->run(assembly, istep + 1);
      }
    }

    specfem::solver::impl::log_time_marching_progress(istep, nstep);
  }

  for (const auto &task : tasks) {
    if (task && !task->should_run(nstep) && task->should_run(-1)) {
      task->run(assembly, nstep);
    }
  }

  for (const auto &task : tasks) {
    task->finalize(assembly);
  }

  specfem::Logger::info(" -- Simulation complete. -- \n");

  return;
}
