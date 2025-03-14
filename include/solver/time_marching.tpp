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
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;

  // Calls to compute mass matrix and invert mass matrix
  this->kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {

    // Predictor phase forward
    this->time_scheme->apply_predictor_phase_forward(acoustic);
    this->time_scheme->apply_predictor_phase_forward(elastic_sv);
    this->time_scheme->apply_predictor_phase_forward(elastic_sh);

    // Update acoustic wavefield:
    // coupling, source interaction, stiffness, divide by mass matrix
    this->kernels.template update_wavefields<acoustic>(istep);

    // Corrector phase forward for acoustic
    this->time_scheme->apply_corrector_phase_forward(acoustic);

    // Update wavefields for elastic wavefields:
    // coupling, source, stiffness, divide by mass matrix
    this->kernels.template update_wavefields<elastic_sv>(istep);
    this->kernels.template update_wavefields<elastic_sh>(istep);

    // Corrector phase forward for elastic
    this->time_scheme->apply_corrector_phase_forward(elastic_sv);
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
  }

  std::cout << std::endl;

  return;
}

template <specfem::dimension::type DimensionType, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionType, NGLL>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
  constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;

  adjoint_kernels.initialize(time_scheme->get_timestep());
  backward_kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  for (const auto [istep, dt] : time_scheme->iterate_backward()) {
    // Adjoint time step
    time_scheme->apply_predictor_phase_forward(acoustic);
    time_scheme->apply_predictor_phase_forward(elastic_sv);
    time_scheme->apply_predictor_phase_forward(elastic_sh);

    adjoint_kernels.template update_wavefields<acoustic>(istep);
    time_scheme->apply_corrector_phase_forward(acoustic);

    adjoint_kernels.template update_wavefields<elastic_sv>(istep);
    adjoint_kernels.template update_wavefields<elastic_sh>(istep);
    time_scheme->apply_corrector_phase_forward(elastic_sv);
    time_scheme->apply_corrector_phase_forward(elastic_sh);

    // Backward time step
    time_scheme->apply_predictor_phase_backward(elastic_sv);
    time_scheme->apply_predictor_phase_backward(acoustic);

    backward_kernels.template update_wavefields<elastic_sv>(istep);
    backward_kernels.template update_wavefields<elastic_sh>(istep);
    time_scheme->apply_corrector_phase_backward(elastic_sv);
    time_scheme->apply_corrector_phase_backward(elastic_sh);

    backward_kernels.template update_wavefields<acoustic>(istep);
    time_scheme->apply_corrector_phase_backward(acoustic);

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
  }

  std::cout << std::endl;

  return;
}

#endif
