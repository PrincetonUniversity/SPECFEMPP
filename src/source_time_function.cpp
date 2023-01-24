#include "../include/source_time_function.h"
#include "../include/config.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <ostream>

KOKKOS_FUNCTION
type_real gaussian(type_real t, type_real f0) {
  // Gaussian wavelet i.e. second integral of a Ricker wavelet

  type_real pi = 2 * std::acos(0.0);

  type_real a = pi * pi * f0 * f0;
  type_real gaussian = -1.0 * std::exp(-a * t * t) / (2.0 * a);

  return gaussian;
}

KOKKOS_FUNCTION
type_real d2gaussian(type_real t, type_real f0) {
  type_real pi = 2 * std::acos(0.0);

  type_real a = pi * pi * f0 * f0;
  type_real d2gaussian = (1.0 - 2.0 * a * t * t) * std::exp(-a * t * t);

  return d2gaussian;
}

KOKKOS_FUNCTION
specfem::forcing_function::Dirac::Dirac(type_real f0, type_real tshift,
                                        type_real factor,
                                        bool use_trick_for_better_pressure)
    : f0(f0), factor(factor),
      use_trick_for_better_pressure(use_trick_for_better_pressure) {

  type_real hdur = 1.0 / this->f0;

  this->t0 = 1.2 * hdur + this->tshift;
}

KOKKOS_FUNCTION
type_real specfem::forcing_function::Dirac::compute(type_real t) {

  type_real val;

  if (this->use_trick_for_better_pressure) {
    val = -1.0 * this->factor * d2gaussian(t - this->tshift, this->f0);
  } else {
    val = -1.0 * this->factor * gaussian(t - this->tshift, this->f0);
  }

  return val;
}

// void specfem::forcing_function::stf::print(std::ostream &out) const {

//   out << "  Error allocating source time function. Base class being called";
//   throw std::runtime_error(
//       "Error allocating source time function. Base class being called");
// }

// void specfem::forcing_function::Dirac::print(std::ostream &out) const {

//   out << "  Forcing function: Dirac force\n"
//       << "                    time shift = " << this->tshift << "\n"
//       << "                    t0 = " << this->t0 << "\n"
//       << "                    f0 = " << this->f0;
// }

// std::ostream &specfem::forcing_function::operator<<(
//     std::ostream &out, const specfem::forcing_function::stf &stf) {

//   stf.print(out);

//   return out;
// };
