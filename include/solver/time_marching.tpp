#ifndef _SPECFEM_SOLVER_TIME_MARCHING_TPP
#define _SPECFEM_SOLVER_TIME_MARCHING_TPP

#include "solver.hpp"
#include "time_marching.hpp"
#include "timescheme/newmark.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::forward,
                                    DimensionType, NGLL>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;

  // Calls to compute mass matrix and invert mass matrix
  this->kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      2 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated = assembly.get_total_number_of_elements();

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    int dofs_updated = 0;
    int elements_updated = 0;

    // Predictor phase forward
    dofs_updated += this->time_scheme->apply_predictor_phase_forward(acoustic);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic_psv);
    dofs_updated +=
        this->time_scheme->apply_predictor_phase_forward(elastic_sh);

    // Update acoustic wavefield:
    // coupling, source interaction, stiffness, divide by mass matrix
    elements_updated += this->kernels.template update_wavefields<acoustic>(istep);

    // Corrector phase forward for acoustic
    dofs_updated += this->time_scheme->apply_corrector_phase_forward(acoustic);

    // Update wavefields for elastic wavefields:
    // coupling, source, stiffness, divide by mass matrix
    elements_updated += this->kernels.template update_wavefields<elastic_psv>(istep);
    elements_updated += this->kernels.template update_wavefields<elastic_sh>(istep);

    // Corrector phase forward for elastic
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated +=
        this->time_scheme->apply_corrector_phase_forward(elastic_sh);

    // Compute seismograms if required
    if (time_scheme->compute_seismogram(istep)) {
      this->kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }
    // Run periodic tasks such as plotting, etc.
    for (const auto &task : tasks) {
      if (task && task->should_run(istep)) {
        task->run();
      }
    }

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
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

  std::cout << std::endl;

  return;
}

template <specfem::dimension::type DimensionType, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionType, NGLL>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;

  adjoint_kernels.initialize(time_scheme->get_timestep());
  backward_kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      4 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated = 2 * assembly.get_total_number_of_elements();

  for (const auto [istep, dt] : time_scheme->iterate_backward()) {
    int dofs_updated = 0;
    int elements_updated = 0;
    // Adjoint time step
    dofs_updated += time_scheme->apply_predictor_phase_forward(acoustic);
    dofs_updated += time_scheme->apply_predictor_phase_forward(elastic_psv);
    dofs_updated += time_scheme->apply_predictor_phase_forward(elastic_sh);

    elements_updated += adjoint_kernels.template update_wavefields<acoustic>(istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(acoustic);

    elements_updated += adjoint_kernels.template update_wavefields<elastic_psv>(istep);
    elements_updated += adjoint_kernels.template update_wavefields<elastic_sh>(istep);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_forward(elastic_sh);

    // Backward time step
    dofs_updated += time_scheme->apply_predictor_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_predictor_phase_backward(acoustic);

    elements_updated += backward_kernels.template update_wavefields<elastic_psv>(istep);
    elements_updated += backward_kernels.template update_wavefields<elastic_sh>(istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_psv);
    dofs_updated += time_scheme->apply_corrector_phase_backward(elastic_sh);

    elements_updated += backward_kernels.template update_wavefields<acoustic>(istep);
    dofs_updated += time_scheme->apply_corrector_phase_backward(acoustic);

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == nstep - 1) {
      specfem::compute::deep_copy(assembly.fields.backward,
                                  assembly.fields.buffer);
    }

    frechet_kernels.compute_derivatives(dt);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      backward_kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }

    for (const auto &task : tasks) {
      if (task && task->should_run(istep)) {
        task->run();
      }
    }

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
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

  std::cout << std::endl;

  return;
}

#endif
