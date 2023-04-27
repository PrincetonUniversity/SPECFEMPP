#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities.cpp"
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_FUNCTION
specfem::forcing_function::Dirac::Dirac(type_real f0, type_real tshift,
                                        type_real factor,
                                        bool use_trick_for_better_pressure)
    : f0(f0), factor(factor), tshift(tshift),
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
