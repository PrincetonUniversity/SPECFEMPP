#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities.cpp"
#include <Kokkos_Core.hpp>
#include <cmath>

specfem::forcing_function::dGaussian::dGaussian(
    const int nsteps, const type_real dt, const type_real f0,
    const type_real tshift, const type_real factor,
    bool use_trick_for_better_pressure)
    : __nsteps(nsteps), __dt(dt), __f0(f0), __factor(factor), __tshift(tshift),
      __use_trick_for_better_pressure(use_trick_for_better_pressure) {

  type_real hdur = 1.0 / this->__f0;

  this->__t0 = -1.2 * hdur + this->__tshift;
}

specfem::forcing_function::dGaussian::dGaussian(
    YAML::Node &dGaussian, const int nsteps, const type_real dt,
    const bool use_trick_for_better_pressure) {
  type_real f0 = dGaussian["f0"].as<type_real>();
  type_real tshift = [dGaussian]() -> type_real {
    if (dGaussian["tshift"]) {
      return dGaussian["tshift"].as<type_real>();
    } else {
      return 0.0;
    }
  }();
  type_real factor = dGaussian["factor"].as<type_real>();

  *this = specfem::forcing_function::dGaussian(nsteps, dt, f0, tshift, factor,
                                               use_trick_for_better_pressure);
}

type_real specfem::forcing_function::dGaussian::compute(type_real t) {

  type_real val;

  if (this->__use_trick_for_better_pressure) {
    val = -1.0 * this->__factor * d3gaussian(t - this->__tshift, this->__f0);
  } else {
    val = -1.0 * this->__factor * d1gaussian(t - this->__tshift, this->__f0);
  }

  return val;
}

void specfem::forcing_function::dGaussian::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    specfem::kokkos::HostView2d<type_real> source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  for (int i = 0; i < nsteps; i++) {
    for (int icomp = 0; icomp < ncomponents; ++icomp) {
      source_time_function(i, icomp) = this->compute(t0 + i * dt);
    }
  }
}

std::string specfem::forcing_function::dGaussian::print() const {
  std::stringstream ss;
  ss << "        dGaussian source time function:\n"
     << "          f0: " << this->__f0 << "\n"
     << "          tshift: " << this->__tshift << "\n"
     << "          factor: " << this->__factor << "\n"
     << "          t0: " << this->__t0 << "\n"
     << "          use_trick_for_better_pressure: "
     << this->__use_trick_for_better_pressure << "\n";

  return ss.str();
}
