#pragma once

#include "specfem/solver/time_marching.hpp"
#include "specfem/solver/impl/update_medium.hpp"
#include "specfem/timescheme/newmark.hpp"
#include "specfem/compute.tpp"
#include "specfem/logger.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::forward,
                                    DimensionTag, NGLL>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;
  constexpr auto poroelastic = specfem::element::medium_tag::poroelastic;
  constexpr auto elastic_psv_t = specfem::element::medium_tag::elastic_psv_t;
  constexpr auto forward = specfem::simulation::field_type::forward;

  // Compute and invert the mass matrix. Media with a cross-rank buffer
  // (elastic, acoustic) overlap the exchange with computation; the rest are
  // local no-ops on the exchange.
  const auto mass_dt = time_scheme->get_timestep();
  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<DimensionTag>{} *
          MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic,
                     elastic_psv_t),
      [&]<typename ElementTags>() {
        specfem::solver::impl::init_medium_mass<
            NGLL, specfem::tags::expand<ElementTags, forward>>(
            assembly, mpi_buffers, mass_dt);
      });

  // The mass-matrix buffers are no longer needed after assembly + inversion.
  mpi_buffers
      .template reset<specfem::data_access::DataClassType::mass_matrix>();

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      2 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated =
      assembly.get_total_number_of_elements();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    int dofs_updated = 0;
    int elements_updated = 0;

    // Predictor phase forward
    dofs_updated += this->time_scheme->apply_predictor_phase_forward(acoustic);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic_psv);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic_sh);
    dofs_updated += this->time_scheme->apply_predictor_phase_forward(poroelastic);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic_psv_t);
    // Update acoustic wavefield: coupling, source, stiffness (outer/inner with
    // overlapped MPI exchange), divide by mass matrix
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, acoustic>>(
        assembly, mpi_buffers, istep);

    // Corrector phase forward for acoustic
    dofs_updated += this->time_scheme->apply_corrector_phase_forward(acoustic);

    // Update wavefields for elastic wavefields
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, elastic>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, elastic_psv>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, elastic_sh>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, elastic_psv_t>>(
        assembly, mpi_buffers, istep);
    // Corrector phase forward for elastic
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_sh);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_psv_t);

    // Update wavefields for poroelastic wavefields
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, forward, poroelastic>>(
        assembly, mpi_buffers, istep);

    // Corrector phase forward for poroelastic
    dofs_updated += this->time_scheme->apply_corrector_phase_forward(poroelastic);

    // Compute seismograms if required
    if (time_scheme->compute_seismogram(istep)) {
      specfem::compute::compute_seismograms<NGLL, specfem::tags::Tags<DimensionTag, forward>>(assembly, time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }
    // Run periodic tasks such as plotting, etc.
    for (const auto &task : tasks) {
      if (task && task->should_run(istep+1)) {
        task->run(assembly, istep+1);
      }
    }

    if (istep % 10 == 0) {
      std::ostringstream message;
      message << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
      specfem::Logger::info(message.str());
    }
    if (dofs_updated != total_dof_to_be_updated) {
      std::ostringstream message;
      message << "The time loop has not updated all the degrees of freedom. "
              << "Only " << dofs_updated << " out of "
              << total_dof_to_be_updated
              << " degrees of freedom have been updated.";

      throw std::runtime_error(message.str());
    }

    if (elements_updated != total_elements_to_be_updated) {
      std::ostringstream message;
      message << "The time loop has not updated all the elements. "
              << "Only " << elements_updated << " out of "
              << total_elements_to_be_updated
              << " elements have been updated.";

      throw std::runtime_error(message.str());
    }
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

template <specfem::element::dimension_tag DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionTag, NGLL>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;
  constexpr auto poroelastic = specfem::element::medium_tag::poroelastic;
  constexpr auto adjoint = specfem::simulation::field_type::adjoint;
  constexpr auto backward = specfem::simulation::field_type::backward;

  // Compute and invert the mass matrix for both wavefields. Media with a
  // cross-rank buffer (elastic, acoustic) overlap the exchange with
  // computation; the rest are local no-ops on the exchange.
  {
    const auto mass_dt = time_scheme->get_timestep();
    const auto mass_set =
        specfem::tag_dispatch::dimension_set<DimensionTag>{} *
        MEDIUM_SET(acoustic, elastic, elastic_psv, elastic_sh, poroelastic);
    const auto init_mass = [&]<specfem::simulation::field_type WF>() {
      specfem::tag_dispatch::for_each(
          mass_set, [&]<typename ElementTags>() {
            specfem::solver::impl::init_medium_mass<
                NGLL, specfem::tags::expand<ElementTags, WF>>(
                assembly, mpi_buffers, mass_dt);
          });
    };
    init_mass.template operator()<adjoint>();
    init_mass.template operator()<backward>();

    // The mass-matrix buffers are no longer needed after assembly + inversion.
    mpi_buffers
        .template reset<specfem::data_access::DataClassType::mass_matrix>();
  }

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      4 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated = 2 * assembly.get_total_number_of_elements();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto &task : tasks) {
    if (task && !task->should_run(nstep) && task->should_run(-1)) {
      task->run(assembly, nstep);
    }
  }

  for (const auto [istep, dt] : time_scheme->iterate_backward()) {
    for (const auto &task : tasks) {
      if (task && task->should_run(istep+1)) {
        task->run(assembly, istep+1);
      }
    }

    int dofs_updated = 0;
    int elements_updated = 0;
    // Adjoint time step
    dofs_updated += time_scheme->apply_predictor_phase_forward(acoustic);
    dofs_updated += time_scheme->apply_predictor_phase_forward(elastic);
    dofs_updated += time_scheme->apply_predictor_phase_forward(elastic_psv);
    dofs_updated += time_scheme->apply_predictor_phase_forward(elastic_sh);
    dofs_updated += time_scheme->apply_predictor_phase_forward(poroelastic);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, adjoint, acoustic>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(acoustic);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, adjoint, elastic>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, adjoint, elastic_psv>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, adjoint, elastic_sh>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_sh);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, adjoint, poroelastic>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(poroelastic);

    // Backward time step
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic);
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic_sh);
    dofs_updated += time_scheme->apply_predictor_phase_backward(acoustic);
    dofs_updated += time_scheme->apply_predictor_phase_backward(poroelastic);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, backward, elastic>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, backward, elastic_psv>>(
        assembly, mpi_buffers, istep);
    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, backward, elastic_sh>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_sh);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, backward, acoustic>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(acoustic);

    elements_updated += specfem::solver::impl::update_medium<
        NGLL, specfem::tags::Tags<DimensionTag, backward, poroelastic>>(
        assembly, mpi_buffers, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(poroelastic);

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == nstep - 1) {
      specfem::assembly::deep_copy(assembly.fields.backward,
                                  assembly.fields.buffer);
    }

    specfem::compute::compute_derivatives<NGLL, specfem::tags::Tags<DimensionTag>>(assembly, dt);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      specfem::compute::compute_seismograms<NGLL, specfem::tags::Tags<DimensionTag, backward>>(assembly, time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }

    if (istep % 10 == 0) {
      std::ostringstream message;
      message << "Progress : executed " << istep << " steps of " << nstep
                << " steps";
      specfem::Logger::info(message.str());
    }

    if (dofs_updated != total_dof_to_be_updated) {
      std::ostringstream message;
      message << "The time loop has not updated all the degrees of freedom. "
              << "Only " << dofs_updated << " out of "
              << total_dof_to_be_updated
              << " degrees of freedom have been updated.";

      throw std::runtime_error(message.str());
    }

    if (elements_updated != total_elements_to_be_updated) {
      std::ostringstream message;
      message << "The time loop has not updated all the elements. "
              << "Only " << elements_updated << " out of "
              << total_elements_to_be_updated
              << " elements have been updated.";

      throw std::runtime_error(message.str());
    }
  }

  for (const auto &task : tasks) {
    task->finalize(assembly);
  }

  specfem::Logger::info(" -- Simulation complete. -- \n");

  return;
}
