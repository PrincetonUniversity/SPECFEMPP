#ifndef _TIME_MARCHING_TPP
#define _TIME_MARCHING_TPP

#include "domain/interface.hpp"
#include "solver.hpp"
#include "time_marching.hpp"
#include "timescheme/interface.hpp"
#include <Kokkos_Core.hpp>

template <typename qp_type>
void specfem::solver::time_marching<qp_type>::run() {

  auto *it = this->it;
  auto elastic_domain = this->elastic_domain;

  const int nstep = it->get_max_timestep();
  auto field = elastic_domain.get_field();
  auto field_dot = elastic_domain.get_field_dot();
  auto field_dot_dot = elastic_domain.get_field_dot_dot();

  while (it->status()) {
    int istep = it->get_timestep();

    type_real timeval = it->get_time();

    Kokkos::Profiling::pushRegion("Stiffness calculation");
    it->apply_predictor_phase(field, field_dot, field_dot_dot);

    elastic_domain.compute_stiffness_interaction();
    elastic_domain.compute_source_interaction(timeval);
    elastic_domain.divide_mass_matrix();

    acoustic_domain.compute_stiffness_interaction();
    acoustic_domain.compute_source_interaction(timeval);
    acoustic_domain.divide_mass_matrix();

    it->apply_corrector_phase(field, field_dot, field_dot_dot);

    if (it->compute_seismogram()) {
      int isig_step = it->get_seismogram_step();
      elastic_domain.compute_seismogram(isig_step);
      it->increment_seismogram_step();
    }
    Kokkos::Profiling::popRegion();

    if (istep % 10 == 0) {
      std::cout << "Progress : executed " << istep << " steps of " << nstep
                << " steps" << std::endl;
    }

    it->increment_time();
  }

  std::cout << std::endl;

  return;
}

#endif
