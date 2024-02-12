#ifndef _TIME_MARCHING_TPP
#define _TIME_MARCHING_TPP

#include "domain/interface.hpp"
#include "solver.hpp"
#include "time_marching.hpp"
#include "timescheme/interface.hpp"
#include <Kokkos_Core.hpp>

template <typename qp_type>
void specfem::solver::time_marching<qp_type>::run() {

  // Special contributions to mass matrix inverse in case of Newmark scheme
  if (it->timescheme() == specfem::enums::time_scheme::type::newmark) {
    elastic_domain.template mass_time_contribution<
        specfem::enums::time_scheme::type::newmark>(it->get_time_increment());
    acoustic_domain.template mass_time_contribution<
        specfem::enums::time_scheme::type::newmark>(it->get_time_increment());
  }

  // Compute and store the inverse of mass matrix for faster computations
  elastic_domain.invert_mass_matrix();
  acoustic_domain.invert_mass_matrix();

  const int nstep = it->get_max_timestep();

  auto acoustic_field = forward_field.get_field<acoustic_type>();
  auto elastic_field = forward_field.get_field<elastic_type>();

  while (it->status()) {
    int istep = it->get_timestep();

    type_real timeval = it->get_time();

    Kokkos::Profiling::pushRegion("Stiffness calculation");
    it->apply_predictor_phase(acoustic_field.field, acoustic_field.field_dot,
                              acoustic_field.field_dot_dot);
    it->apply_predictor_phase(elastic_field.field, elastic_field.field_dot,
                              elastic_field.field_dot_dot);

    acoustic_elastic_interface.compute_coupling();
    acoustic_domain.compute_source_interaction(timeval);
    acoustic_domain.compute_stiffness_interaction();
    acoustic_domain.divide_mass_matrix();

    it->apply_corrector_phase(acoustic_field.field, acoustic_field.field_dot,
                              acoustic_field.field_dot_dot);

    elastic_acoustic_interface.compute_coupling();
    elastic_domain.compute_source_interaction(timeval);
    elastic_domain.compute_stiffness_interaction();
    elastic_domain.divide_mass_matrix();

    it->apply_corrector_phase(elastic_field.field, elastic_field.field_dot,
                              elastic_field.field_dot_dot);
    Kokkos::Profiling::popRegion();

    // if (it->compute_seismogram()) {
    //   int isig_step = it->get_seismogram_step();
    //   acoustic_domain.compute_seismogram(isig_step);
    //   elastic_domain.compute_seismogram(isig_step);
    //   it->increment_seismogram_step();
    // }

    it->increment_time();

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
    }
  }

  std::cout << std::endl;

  return;
}

#endif
