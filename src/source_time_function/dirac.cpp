#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities.cpp"
#include <Kokkos_Core.hpp>
#include <cmath>

specfem::forcing_function::Dirac::Dirac(const type_real dt, const type_real f0,
                                        const type_real tshift,
                                        const type_real factor,
                                        bool use_trick_for_better_pressure)
    : dt(dt), f0(f0), factor(factor), tshift(tshift),
      use_trick_for_better_pressure(use_trick_for_better_pressure) {

  type_real hdur = 1.0 / this->f0;

  this->t0 = -1.2 * hdur + this->tshift;
}

specfem::forcing_function::Dirac::Dirac(
    YAML::Node &Dirac, const type_real dt,
    const bool use_trick_for_better_pressure) {
  type_real f0 = 1.0 / (10.0 * dt);
  type_real tshift = Dirac["tshift"].as<type_real>();
  type_real factor = Dirac["factor"].as<type_real>();

  *this = specfem::forcing_function::Dirac(dt, f0, tshift, factor,
                                           use_trick_for_better_pressure);
}

type_real specfem::forcing_function::Dirac::compute(type_real t) {

  type_real val;

  if (this->use_trick_for_better_pressure) {
    val = -1.0 * this->factor * d2gaussian(t - this->tshift, this->f0);
  } else {
    val = -1.0 * this->factor * gaussian(t - this->tshift, this->f0);
  }

  return val;
}

void specfem::forcing_function::Dirac::compute_source_time_function(
    const int nsteps,
    specfem::kokkos::HostView1d<type_real> source_time_function) {

  for (int i = 0; i < nsteps; i++) {
    source_time_function(i) = this->compute(this->t0 + i * this->dt);
  }
}

std::string specfem::forcing_function::Dirac::print() const {
  std::stringstream ss;
  ss << "        Dirac source time function:\n"
     << "          f0: " << this->f0 << "\n"
     << "          tshift: " << this->tshift << "\n"
     << "          factor: " << this->factor << "\n"
     << "          t0: " << this->t0 << "\n"
     << "          use_trick_for_better_pressure: "
     << this->use_trick_for_better_pressure << "\n";

  return ss.str();
}
