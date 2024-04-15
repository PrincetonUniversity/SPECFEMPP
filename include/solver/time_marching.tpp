#ifndef _SPECFEM_SOLVER_TIME_MARCHING_TPP
#define _SPECFEM_SOLVER_TIME_MARCHING_TPP

#include "domain/interface.hpp"
#include "solver.hpp"
#include "time_marching.hpp"
#include "timescheme/interface.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionType, typename qp_type>
void specfem::solver::time_marching<specfem::simulation::type::forward,
                                    DimensionType, qp_type>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;

  kernels.initialize(time_scheme->timescheme(), time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  for (int istep : time_scheme->iterate()) {
    time_scheme->apply_predictor_phase_forward(acoustic);
    time_scheme->apply_predictor_phase_forward(elastic);

    kernels.template update_wavefields<acoustic>(istep);
    time_scheme->apply_corrector_phase_forward(acoustic);

    kernels.template update_wavefields<elastic>(istep);
    time_scheme->apply_corrector_phase_forward(elastic);

    if (time_scheme->compute_seismogram(istep)) {
      kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
    }
  }

  std::cout << std::endl;

  return;
}

template <specfem::dimension::type DimensionType, typename qp_type>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionType, qp_type>::run() {

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;

  adjoint_kernels.initialize(time_scheme->timescheme(),
                             time_scheme->get_timestep());
  backward_kernels.initialize(time_scheme->timescheme(),
                              time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  for (int istep : time_scheme->iterate()) {
    const int backward_step = nstep - istep - 1;
    // Adjoint time step
    time_scheme->apply_predictor_phase_forward(acoustic);
    time_scheme->apply_predictor_phase_forward(elastic);

    adjoint_kernels.template update_wavefields<acoustic>(backward_step);
    time_scheme->apply_corrector_phase_forward(acoustic);

    adjoint_kernels.template update_wavefields<elastic>(backward_step);
    time_scheme->apply_corrector_phase_forward(elastic);

    // Backward time step
    time_scheme->apply_predictor_phase_backward(elastic);
    time_scheme->apply_predictor_phase_backward(acoustic);

    backward_kernels.template update_wavefields<elastic>(backward_step);
    time_scheme->apply_corrector_phase_backward(elastic);

    backward_kernels.template update_wavefields<acoustic>(backward_step);
    time_scheme->apply_corrector_phase_backward(acoustic);

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == 0) {
      specfem::compute::deep_copy(assembly.fields.backward,
                                  assembly.fields.buffer);
    }

    // frechet_kernels.compute_frechet_derivatives<acoustic>(istep);
    // frechet_kernels.compute_frechet_derivatives<elastic>(istep);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      backward_kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
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
