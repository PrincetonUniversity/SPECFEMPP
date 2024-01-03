#include "specfem_setup.hpp"
#include "timescheme/interface.hpp"
#include <ostream>

specfem::TimeScheme::Newmark::Newmark(const int nstep, const type_real t0,
                                      const type_real dt,
                                      const int nstep_between_samples)
    : nstep(nstep), t0(t0), deltat(dt) {
  this->deltatover2 = dt / 2.0;
  this->deltatsquareover2 = deltatover2 * dt;
  this->current_time = this->t0;
  this->nstep_between_samples = nstep_between_samples;
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

void specfem::TimeScheme::Newmark::apply_predictor_phase(
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        field_dot_dot) {

  const int nglob = field.extent(0);
  const int components = field.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_predictor_phase",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        // update displacements
        field(iglob, idim) +=
            this->deltat * field_dot(iglob, idim) +
            this->deltatsquareover2 * field_dot_dot(iglob, idim);
        // apply predictor phase
        field_dot(iglob, idim) +=
            this->deltatover2 * field_dot_dot(iglob, idim);
        // reset acceleration
        field_dot_dot(iglob, idim) = 0;
      });

  return;
}

void specfem::TimeScheme::Newmark::apply_corrector_phase(
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        field_dot_dot) {

  const int nglob = field_dot.extent(0);
  const int components = field_dot.extent(1);

  Kokkos::parallel_for(
      "specfem::TimeScheme::Newmark::apply_corrector_phase",
      specfem::kokkos::DeviceRange(0, components * nglob),
      KOKKOS_CLASS_LAMBDA(const int in) {
        const int iglob = in % nglob;
        const int idim = in / nglob;
        type_real acceleration = field_dot_dot(iglob, idim);
        type_real velocity = field_dot(iglob, idim);
        // apply corrector phase
        field_dot(iglob, idim) +=
            this->deltatover2 * field_dot_dot(iglob, idim);
      });

  return;
}

void specfem::TimeScheme::Newmark::print(std::ostream &message) const {
  message << "  Time Scheme:\n"
          << "------------------------------\n"
          << "- Newmark\n"
          << "    dt = " << this->deltat << "\n"
          << "    number of time steps = " << this->nstep << "\n"
          << "    Start time = " << this->t0 << "\n";
}
