#include "../include/timescheme.h"
#include "../include/config.h"
#include <ostream>

specfem::TimeScheme::Newmark::Newmark(int nstep, type_real t0, type_real dt)
    : nstep(nstep), t0(t0), deltat(dt) {
  this->deltatover2 = dt / 2.0;
  this->deltatsquareover2 = deltatover2 * dt;
  this->current_time = this->t0;
}

void specfem::TimeScheme::Newmark::increment_time() {
  this->istep++;
  this->current_time += this->deltat;
  return;
}

void specfem::TimeScheme::Newmark::reset_time() {
  this->istep = 0;
  this->current_time = this->t0;
  return;
}

KOKKOS_IMPL_HOST_FUNCTION
void specfem::TimeScheme::Newmark::apply_predictor_phase(
    const specfem::Domain::Domain *domain) {
  auto field = domain->get_field();
  auto field_dot = domain->get_field_dot();
  auto field_dot_dot = domain->get_field_dot_dot();

  const int nglob = field.extent(0);
  const int ndim = field.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_predictor_phase",
      specfem::kokkos::DeviceRange(0, nglob),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        for (int idim = 0; idim < ndim; idim++) {
          // update displacements
          field(iglob, idim) +=
              this->deltat * field_dot(iglob, idim) +
              this->deltatsquareover2 * field_dot_dot(iglob, idim);
          // apply predictor phase
          field_dot(iglob, idim) +=
              this->deltatover2 * field_dot_dot(iglob, idim);
          // reset acceleration
          field_dot_dot(iglob, idim) = 0;
        }
      });

  return;
}

KOKKOS_IMPL_HOST_FUNCTION
void specfem::TimeScheme::Newmark::apply_corrector_phase(
    const specfem::Domain::Domain *domain) {

  auto field_dot = domain->get_field_dot();
  auto field_dot_dot = domain->get_field_dot_dot();

  const int nglob = field_dot.extent(0);
  const int ndim = field_dot.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_predictor_phase",
      specfem::kokkos::DeviceRange(0, nglob),
      KOKKOS_CLASS_LAMBDA(const int iglob) {
        for (int idim = 0; idim < ndim; idim++) {
          // apply corrector phase
          field_dot(iglob, idim) +=
              this->deltatover2 * field_dot_dot(iglob, idim);
        }
      });

  return;
}

void specfem::TimeScheme::TimeScheme::print(std::ostream &out) const {
  out << "Time scheme wasn't initialized properly. Base class being called";

  throw std::runtime_error(
      "Time scheme wasn't initialized properly. Base class being called");
}

void specfem::TimeScheme::Newmark::print(std::ostream &message) const {
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    dt = " << this->deltat << "\n"
          << "    number of time steps = " << this->nstep << "\n"
          << "    Start time = " << this->t0 << "\n";
}

std::ostream &
specfem::TimeScheme::operator<<(std::ostream &out,
                                specfem::TimeScheme::TimeScheme &ts) {
  ts.print(out);

  return out;
}
