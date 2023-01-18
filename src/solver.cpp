#include "../include/solver.h"
#include "../include/domain.h"
#include "../include/timescheme.h"

void specfem::solver::time_marching::run() {

  specfem::TimeScheme::TimeScheme *it = this->it;
  specfem::Domain::Domain *domain = this->domain;

  while (it->status()) {
    int istep = it->get_timestep();

    type_real timeval = it->get_time();

    it->apply_predictor_phase(domain);

    domain->compute_stiffness_interaction();
    domain->compute_source_interaction(timeval);
    domain->divide_mass_matrix();

    it->apply_corrector_phase(domain);

    it->increment_time();
  }

  return;
}
