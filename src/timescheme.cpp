#include "../include/timescheme.h"
#include "../include/config.h"
#include <ostream>

specfem::TimeScheme::Newmark::Newmark(int nstep, type_real t0, type_real dt)
    : nstep(nstep), t0(t0), deltat(dt) {
  this->deltatover2 = dt / 2.0;
  this->deltatsquareover2 = deltatover2 * dt;
  this->current_time = this->t0;
}

void specfem::TimeScheme::Newmark::update_time() {
  this->istep++;
  this->current_time += this->deltat;
  return;
}

void specfem::TimeScheme::Newmark::reset_time() {
  this->istep = 0;
  this->current_time = this->t0;
  return;
}

void specfem::TimeScheme::TimeScheme::print(std::ostream &out) const {

  out << "Time scheme wasn't initialized properly. Base class being called";

  throw std::runtime_error(
      "Time scheme wasn't initialized properly. Base class being called");
}

void specfem::TimeScheme::Newmark::print(std::ostream &out) const {

  out << "  Time Scheme : Newmark\n"
      << "                dt = " << this->deltat << "\n"
      << "                number of time steps = " << this->nstep << "\n"
      << "                Start time = " << this->t0;
}

std::ostream &
specfem::TimeScheme::operator<<(std::ostream &out,
                                specfem::TimeScheme::TimeScheme &ts) {

  ts.print(out);

  return out;
}
