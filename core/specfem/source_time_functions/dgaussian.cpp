#include "dgaussian.hpp"
#include "impl/time_functions.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

specfem::source_time_functions::dGaussian::dGaussian(
    const int nsteps, const type_real dt, const type_real f0,
    const type_real tshift, const type_real factor,
    bool use_trick_for_better_pressure, const type_real t0_factor)
    : nsteps_(nsteps), dt_(dt), f0_(f0), factor_(factor), tshift_(tshift),
      t0_factor_(t0_factor),
      use_trick_for_better_pressure_(use_trick_for_better_pressure) {

  type_real hdur = 1.0 / this->f0_;

  // Default t0_factor is 1.2 for dGaussian, see header file
  this->t0_ = -this->t0_factor_ * hdur + this->tshift_;
}

specfem::source_time_functions::dGaussian::dGaussian(
    YAML::Node &dGaussian, const int nsteps, const type_real dt,
    const bool use_trick_for_better_pressure, const type_real t0_factor) {
  type_real f0 = dGaussian["f0"].as<type_real>();
  type_real tshift = [dGaussian]() -> type_real {
    if (dGaussian["tshift"]) {
      return dGaussian["tshift"].as<type_real>();
    } else {
      return 0.0;
    }
  }();
  type_real factor = dGaussian["factor"].as<type_real>();

  *this = specfem::source_time_functions::dGaussian(
      nsteps, dt, f0, tshift, factor, use_trick_for_better_pressure, t0_factor);
}

type_real specfem::source_time_functions::dGaussian::compute(type_real t) {

  type_real val;

  if (this->use_trick_for_better_pressure_) {
    val = this->factor_ * specfem::source_time_functions::impl::d3gaussian(
                              t - this->tshift_, this->f0_);
  } else {
    val = this->factor_ * specfem::source_time_functions::impl::d1gaussian(
                              t - this->tshift_, this->f0_);
  }

  return val;
}

void specfem::source_time_functions::dGaussian::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    specfem::kokkos::HostView2d<type_real> source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  for (int i = 0; i < nsteps; i++) {
    for (int icomp = 0; icomp < ncomponents; ++icomp) {
      source_time_function(i, icomp) = this->compute(t0 + i * dt);
    }
  }
}

std::string specfem::source_time_functions::dGaussian::print() const {
  std::stringstream ss;
  ss << "        dGaussian source time function:\n"
     << "          f0: " << this->f0_ << "\n"
     << "          tshift: " << this->tshift_ << "\n"
     << "          factor: " << this->factor_ << "\n"
     << "          t0: " << this->t0_ << "\n"
     << "          use_trick_for_better_pressure: "
     << this->use_trick_for_better_pressure_ << "\n";

  return ss.str();
}

bool specfem::source_time_functions::dGaussian::operator==(
    const stf &other) const {

  // First check base class equality
  if (!specfem::source_time_functions::stf::operator==(other))
    return false;

  // Then check if the other object is a dGaussian
  auto other_dgaussian =
      dynamic_cast<const specfem::source_time_functions::dGaussian *>(&other);

  // Check if cast was successful
  if (!other_dgaussian)
    return false;

  return (specfem::utilities::is_close(this->t0_, other_dgaussian->get_t0()) &&
          specfem::utilities::is_close(this->dt_, other_dgaussian->get_dt()) &&
          specfem::utilities::is_close(this->f0_, other_dgaussian->get_f0()) &&
          this->nsteps_ == other_dgaussian->get_nsteps() &&
          specfem::utilities::is_close(this->tshift_,
                                       other_dgaussian->get_tshift()) &&
          specfem::utilities::is_close(this->factor_,
                                       other_dgaussian->get_factor()) &&
          this->use_trick_for_better_pressure_ ==
              other_dgaussian->get_use_trick_for_better_pressure());
}

bool specfem::source_time_functions::dGaussian::operator!=(
    const stf &other) const {
  return !(*(this) == other);
}
