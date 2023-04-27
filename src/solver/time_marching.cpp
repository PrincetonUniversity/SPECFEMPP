#include "domain/interface.hpp"
#include "solver/interface.hpp"
#include "timescheme/interface.hpp"
#include <Kokkos_Core.hpp>

void specfem::solver::time_marching::run() {

  specfem::TimeScheme::TimeScheme *it = this->it;
  specfem::Domain::Domain *domain = this->domain;

  const int nstep = it->get_max_timestep();

  while (it->status()) {
    int istep = it->get_timestep();

    type_real timeval = it->get_time();

    Kokkos::Profiling::pushRegion("Stiffness calculation");
    it->apply_predictor_phase(domain);

    domain->compute_stiffness_interaction_calling_routine();
    domain->compute_source_interaction(timeval);
    domain->divide_mass_matrix();

    it->apply_corrector_phase(domain);

    if (it->compute_seismogram()) {
      int isig_step = it->get_seismogram_step();
      domain->compute_seismogram(isig_step);
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
