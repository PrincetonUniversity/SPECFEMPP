#include "gaussianhdur.hpp"
#include "constants.hpp"
#include "impl/time_functions.hpp"
#include "specfem/logger.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

specfem::source_time_functions::GaussianHdur::GaussianHdur(
    const int nsteps, const type_real dt, const type_real hdur,
    const type_real tshift, const type_real factor,
    bool use_trick_for_better_pressure, const type_real t0_factor)
    : nsteps_(nsteps), dt_(dt), hdur_(hdur), factor_(factor), tshift_(tshift),
      use_trick_for_better_pressure_(use_trick_for_better_pressure),
      t0_factor_(t0_factor) {

  // Corrects the half duration if it's too small compared to the time step
  if (this->hdur_ < 5.0 * this->dt_) {
    this->hdur_ = 5.0 * this->dt_;
    specfem::Logger::warning(
        "GaussianHdur half duration hdur is too small compared to dt. "
        "Setting hdur = 5 * dt = " +
        std::to_string(this->hdur_));
  }

  // SPECFEM3D Cartesian computes the starttime before correction of the half
  // duration.
  // Default t0_factor is 1.5 for GaussianHdur, see header file
  // This is adopted from SPECFEM3D Cartesian
  this->t0_ = -this->t0_factor_ * this->hdur_ + this->tshift_;

  // The half duration is then based on the empirical relation to of a
  // triangular source time function
  this->hdur_ =
      this->hdur_ / specfem::constants::empirical::SOURCE_DECAY_MIMIC_TRIANGLE;
}

specfem::source_time_functions::GaussianHdur::GaussianHdur(
    YAML::Node &GaussianHdurNode, const int nsteps, const type_real dt,
    const bool use_trick_for_better_pressure, const type_real t0_factor) {

  type_real hdur = GaussianHdurNode["hdur"].as<type_real>();

  type_real tshift = [GaussianHdurNode]() -> type_real {
    if (GaussianHdurNode["tshift"]) {
      return GaussianHdurNode["tshift"].as<type_real>();
    } else {
      return 0.0;
    }
  }();
  type_real factor = GaussianHdurNode["factor"].as<type_real>();

  *this = specfem::source_time_functions::GaussianHdur(
      nsteps, dt, hdur, tshift, factor, use_trick_for_better_pressure,
      t0_factor);
}

type_real specfem::source_time_functions::GaussianHdur::compute(type_real t) {

  type_real val;

  if (this->use_trick_for_better_pressure_) {
    val = this->factor_ * specfem::source_time_functions::impl::d2gaussian_hdur(
                              t - this->tshift_, this->hdur_);
  } else {
    val = this->factor_ * specfem::source_time_functions::impl::gaussian_hdur(
                              t - this->tshift_, this->hdur_);
  }

  return val;
}

void specfem::source_time_functions::GaussianHdur::compute_source_time_function(
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

std::string specfem::source_time_functions::GaussianHdur::print() const {
  std::stringstream ss;
  ss << "        GaussianHdur source time function:\n"
     << "          hdur: " << this->hdur_ << "\n"
     << "          tshift: " << this->tshift_ << "\n"
     << "          factor: " << this->factor_ << "\n"
     << "          t0: " << this->t0_ << "\n"
     << "          use_trick_for_better_pressure: "
     << this->use_trick_for_better_pressure_ << "\n";

  return ss.str();
}

bool specfem::source_time_functions::GaussianHdur::operator==(
    const stf &other) const {
  // First check base class equality
  if (!specfem::source_time_functions::stf::operator==(other))
    return false;

  // Then check if the other object is a GaussianHdur
  auto other_gaussian_hdur =
      dynamic_cast<const specfem::source_time_functions::GaussianHdur *>(
          &other);
  if (!other_gaussian_hdur)
    return false;

  return (specfem::utilities::is_close(this->hdur_,
                                       other_gaussian_hdur->get_hdur()) &&
          specfem::utilities::is_close(this->tshift_,
                                       other_gaussian_hdur->get_tshift()) &&
          specfem::utilities::is_close(this->factor_,
                                       other_gaussian_hdur->get_factor()) &&
          this->use_trick_for_better_pressure_ ==
              other_gaussian_hdur->get_use_trick_for_better_pressure());
};

bool specfem::source_time_functions::GaussianHdur::operator!=(
    const stf &other) const {
  return !(*this == other);
}
