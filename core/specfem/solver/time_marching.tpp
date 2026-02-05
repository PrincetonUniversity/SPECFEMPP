#pragma once

#include "specfem/solver/time_marching.hpp"
#include "specfem/timescheme/newmark.hpp"
#include "specfem/compute.hpp"
#include "specfem/logger.hpp"
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

  // Calls to compute mass matrix and invert mass matrix
  specfem::compute::initialize_mass_matrix<forward, DimensionTag, NGLL>(assembly, time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      2 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated = assembly.get_total_number_of_elements();

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
    // Update acoustic wavefield:
    // coupling, source interaction, stiffness, divide by mass matrix
    elements_updated += specfem::compute::update_wavefields<specfem::simulation::field_type::forward, DimensionTag, NGLL, acoustic>(assembly, istep);

    // Corrector phase forward for acoustic
    dofs_updated += this->time_scheme->apply_corrector_phase_forward(acoustic);

    // Update wavefields for elastic wavefields:
    // coupling, source, stiffness, divide by mass matrix
    elements_updated += specfem::compute::update_wavefields<forward, DimensionTag, NGLL, elastic>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<forward, DimensionTag, NGLL, elastic_psv>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<forward, DimensionTag, NGLL, elastic_sh>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<forward, DimensionTag, NGLL, elastic_psv_t>(assembly, istep);
    // Corrector phase forward for elastic
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_sh);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_psv_t);

    // Update wavefields for poroelastic wavefields:
    // coupling, source, stiffness, divide by mass matrix
    elements_updated += specfem::compute::update_wavefields<forward, DimensionTag, NGLL, poroelastic>(assembly, istep);

    // Corrector phase forward for poroelastic
    dofs_updated += this->time_scheme->apply_corrector_phase_forward(poroelastic);

    // Compute seismograms if required
    if (time_scheme->compute_seismogram(istep)) {
      specfem::compute::compute_seismograms<forward, DimensionTag, NGLL>(assembly, time_scheme->get_seismogram_step());
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

  specfem::compute::initialize_mass_matrix<adjoint, DimensionTag, NGLL>(assembly, time_scheme->get_timestep());
  specfem::compute::initialize_mass_matrix<backward, DimensionTag, NGLL>(assembly, time_scheme->get_timestep());

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

    elements_updated += specfem::compute::update_wavefields<adjoint, DimensionTag, NGLL, acoustic>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(acoustic);

    elements_updated += specfem::compute::update_wavefields<adjoint, DimensionTag, NGLL, elastic>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<adjoint, DimensionTag, NGLL, elastic_psv>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<adjoint, DimensionTag, NGLL, elastic_sh>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_sh);

    elements_updated += specfem::compute::update_wavefields<adjoint, DimensionTag, NGLL, poroelastic>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(poroelastic);

    // Backward time step
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic);
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic_sh);
    dofs_updated += time_scheme->apply_predictor_phase_backward(acoustic);
    dofs_updated += time_scheme->apply_predictor_phase_backward(poroelastic);

    elements_updated += specfem::compute::update_wavefields<backward, DimensionTag, NGLL, elastic>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<backward, DimensionTag, NGLL, elastic_psv>(assembly, istep);
    elements_updated += specfem::compute::update_wavefields<backward, DimensionTag, NGLL, elastic_sh>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_sh);

    elements_updated += specfem::compute::update_wavefields<backward, DimensionTag, NGLL, acoustic>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(acoustic);

    elements_updated += specfem::compute::update_wavefields<backward, DimensionTag, NGLL, poroelastic>(assembly, istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(poroelastic);

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == nstep - 1) {
      specfem::assembly::deep_copy(assembly.fields.backward,
                                  assembly.fields.buffer);
    }

    specfem::compute::compute_derivatives<DimensionTag, NGLL>(assembly, dt);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      specfem::compute::compute_seismograms<backward, DimensionTag, NGLL>(assembly, time_scheme->get_seismogram_step());
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
