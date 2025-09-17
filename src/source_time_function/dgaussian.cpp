#include "source_time_function/impl/time_functions.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
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
    val = this->__factor * specfem::forcing_function::impl::d3gaussian(
                               t - this->__tshift, this->__f0);
  } else {
    val = this->__factor * specfem::forcing_function::impl::d1gaussian(
                               t - this->__tshift, this->__f0);
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

bool specfem::forcing_function::dGaussian::operator==(
    const specfem::forcing_function::stf &other) const {

  // First check base class equality
  if (!specfem::forcing_function::stf::operator==(other))
    return false;

  // Then check if the other object is a dGaussian
  auto other_dgaussian =
      dynamic_cast<const specfem::forcing_function::dGaussian *>(&other);

  // Check if cast was successful
  if (!other_dgaussian)
    return false;

  return (specfem::utilities::is_close(this->__t0, other_dgaussian->get_t0()) &&
          specfem::utilities::is_close(this->__dt, other_dgaussian->get_dt()) &&
          specfem::utilities::is_close(this->__f0, other_dgaussian->get_f0()) &&
          this->__nsteps == other_dgaussian->get_nsteps() &&
          specfem::utilities::is_close(this->__tshift,
                                       other_dgaussian->get_tshift()) &&
          specfem::utilities::is_close(this->__factor,
                                       other_dgaussian->get_factor()) &&
          this->__use_trick_for_better_pressure ==
              other_dgaussian->get_use_trick_for_better_pressure());
}

bool specfem::forcing_function::dGaussian::operator!=(
    const specfem::forcing_function::stf &other) const {
  return !(*(this) == other);
}
