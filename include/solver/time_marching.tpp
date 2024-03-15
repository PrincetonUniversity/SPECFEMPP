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
    // Adjoint time step
    time_scheme->apply_predictor_phase_forward(elastic);
    time_scheme->apply_predictor_phase_forward(acoustic);

    adjoint_kernels.template update_wavefields<elastic>(nstep - istep);
    time_scheme->apply_corrector_phase_forward(elastic);

    adjoint_kernels.template update_wavefields<acoustic>(nstep - istep);
    time_scheme->apply_corrector_phase_forward(acoustic);

    // Backward time step
    time_scheme->apply_predictor_phase_backward(acoustic);
    time_scheme->apply_predictor_phase_backward(elastic);

    backward_kernels.template update_wavefields<acoustic>(nstep - istep);
    time_scheme->apply_corrector_phase_backward(acoustic);

    backward_kernels.template update_wavefields<elastic>(nstep - istep);
    time_scheme->apply_corrector_phase_backward(elastic);

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == 0) {
      specfem::compute::deep_copy(assembly.fields.buffer,
                                  assembly.fields.backward);
    }

    // frechet_kernels.compute_frechet_derivatives<acoustic>(istep);
    // frechet_kernels.compute_frechet_derivatives<elastic>(istep);

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
    }
  }

  std::cout << std::endl;

  return;
}

// template <typename Kernels, typename TimeScheme>
// void specfem::solver::time_marching<specfem::simulation::type::adjoint,
// Kernels,
//                                     TimeScheme>::run() {

//   Kernels::initialize(TimeScheme::timescheme());

//   const int nstep = TimeScheme::get_max_timestep();

//   for (int istep : TimeScheme::iterate()) {
//     TimeScheme::apply_predictor_phase<acoustic>();
//     TimeScheme::apply_predictor_phase<elastic>();

//     Kernels::update_wavefields<acoustic>(istep);
//     TimeScheme::apply_corrector_phase<acoustic>();

//     Kernels::update_wavefields<elastic>(istep);
//     TimeScheme::apply_corrector_phase<elastic>();

//     Kernels::compute_frechlet_derivatives<acoustic>(istep);
//     Kernels::compute_frechlet_derivatives<elastic>(istep);

//     if (istep % 10 == 0) {
//       std::cout << "Progress : executed " << istep << " steps of " << nstep
//                 << " steps" << std::endl;
//     }
//   }

//   std::cout << std::endl;

//   return;
// }

// template <typename qp_type>
// void specfem::solver::time_marching<qp_type>::run() {

//   // Special contributions to mass matrix inverse in case of Newmark scheme
//   if (it->timescheme() == specfem::enums::time_scheme::type::newmark) {
//     elastic_domain.template mass_time_contribution<
//         specfem::enums::time_scheme::type::newmark>(it->get_time_increment());
//     acoustic_domain.template mass_time_contribution<
//         specfem::enums::time_scheme::type::newmark>(it->get_time_increment());
//   }

//   // Compute and store the inverse of mass matrix for faster computations
//   elastic_domain.invert_mass_matrix();
//   acoustic_domain.invert_mass_matrix();

//   const int nstep = it->get_max_timestep();

//   auto acoustic_field = forward_field.get_field<acoustic_type>();
//   auto elastic_field = forward_field.get_field<elastic_type>();

//   while (it->status()) {
//     int istep = it->get_timestep();

//     Kokkos::Profiling::pushRegion("Stiffness calculation");
//     it->apply_predictor_phase(acoustic_field.field, acoustic_field.field_dot,
//                               acoustic_field.field_dot_dot);
//     it->apply_predictor_phase(elastic_field.field, elastic_field.field_dot,
//                               elastic_field.field_dot_dot);

//     acoustic_elastic_interface.compute_coupling();
//     acoustic_domain.compute_source_interaction(istep);
//     acoustic_domain.compute_stiffness_interaction();
//     acoustic_domain.divide_mass_matrix();

//     it->apply_corrector_phase(acoustic_field.field, acoustic_field.field_dot,
//                               acoustic_field.field_dot_dot);

//     elastic_acoustic_interface.compute_coupling();
//     elastic_domain.compute_source_interaction(istep);
//     elastic_domain.compute_stiffness_interaction();
//     elastic_domain.divide_mass_matrix();

//     it->apply_corrector_phase(elastic_field.field, elastic_field.field_dot,
//                               elastic_field.field_dot_dot);
//     Kokkos::Profiling::popRegion();

//     if (it->compute_seismogram()) {
//       int isig_step = it->get_seismogram_step();
//       acoustic_domain.compute_seismograms(isig_step);
//       elastic_domain.compute_seismograms(isig_step);
//       it->increment_seismogram_step();
//     }

//     it->increment_time();

//     if (istep % 10 == 0) {
//       std::cout << "Progress : executed " << istep << " steps of " << nstep
//                 << " steps" << std::endl;
//     }
//   }

//   std::cout << std::endl;

//   return;
// }

#endif
