#include "gaussian.hpp"
#include "constants.hpp"
#include "impl/time_functions.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

specfem::source_time_functions::Gaussian::Gaussian(
    const int nsteps, const type_real dt, const type_real f0,
    const type_real tshift, const type_real factor,
    bool use_trick_for_better_pressure, const type_real t0_factor)
    : nsteps_(nsteps), dt_(dt), f0_(f0), factor_(factor), tshift_(tshift),
      t0_factor_(t0_factor),
      use_trick_for_better_pressure_(use_trick_for_better_pressure) {

  type_real hdur = 1.0 / this->f0_;

  // Default t0_factor is 2.0 for Gaussian, see header file
  this->t0_ = -this->t0_factor_ * hdur + this->tshift_;
}

specfem::source_time_functions::Gaussian::Gaussian(
    YAML::Node &GaussianNode, const int nsteps, const type_real dt,
    const bool use_trick_for_better_pressure, const type_real t0_factor) {

  type_real f0 = GaussianNode["f0"].as<type_real>();

  type_real tshift = [GaussianNode]() -> type_real {
    if (GaussianNode["tshift"]) {
      return GaussianNode["tshift"].as<type_real>();
    } else {
      return 0.0;
    }
  }();
  type_real factor = GaussianNode["factor"].as<type_real>();

  *this = specfem::source_time_functions::Gaussian(
      nsteps, dt, f0, tshift, factor, use_trick_for_better_pressure, t0_factor);
}

type_real specfem::source_time_functions::Gaussian::compute(type_real t) {

  type_real val;

  if (this->use_trick_for_better_pressure_) {
    val = this->factor_ * specfem::source_time_functions::impl::d2gaussian(
                              t - this->tshift_, this->f0_);
  } else {
    val = this->factor_ * specfem::source_time_functions::impl::gaussian(
                              t - this->tshift_, this->f0_);
  }

  return val;
}

void specfem::source_time_functions::Gaussian::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  for (int i = 0; i < nsteps; i++) {
    for (int icomp = 0; icomp < ncomponents; ++icomp) {
      source_time_function(i, icomp) = this->compute(t0 + i * dt);
    }
  }
}

std::string specfem::source_time_functions::Gaussian::print() const {
  std::stringstream ss;
  ss << "        Gaussian source time function:\n"
     << "          f0: " << this->f0_ << "\n"
     << "          tshift: " << this->tshift_ << "\n"
     << "          factor: " << this->factor_ << "\n"
     << "          t0: " << this->t0_ << "\n"
     << "          use_trick_for_better_pressure: "
     << this->use_trick_for_better_pressure_ << "\n";

  return ss.str();
}

bool specfem::source_time_functions::Gaussian::operator==(
    const stf &other) const {
  // First check base class equality
  if (!specfem::source_time_functions::stf::operator==(other))
    return false;

  // Then check if the other object is a dGaussian
  auto other_gaussian =
      dynamic_cast<const specfem::source_time_functions::Gaussian *>(&other);
  if (!other_gaussian)
    return false;

  return (specfem::utilities::is_close(this->f0_, other_gaussian->get_f0()) &&
          specfem::utilities::is_close(this->tshift_,
                                       other_gaussian->get_tshift()) &&
          specfem::utilities::is_close(this->factor_,
                                       other_gaussian->get_factor()) &&
          this->use_trick_for_better_pressure_ ==
              other_gaussian->get_use_trick_for_better_pressure());
};

bool specfem::source_time_functions::Gaussian::operator!=(
    const stf &other) const {
  return !(*this == other);
}
