#ifndef SOURCE_TIME_FUNCTION_H
#define SOURCE_TIME_FUNCTION_H

#include "config.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {

class stf {
public:
  KOKKOS_FUNCTION virtual type_real compute(type_real t) { return 0.0; }
  KOKKOS_FUNCTION virtual void update_tshift(type_real tshift){};
  KOKKOS_FUNCTION virtual type_real get_t0() const { return 0.0; }

  // virtual void print(std::ostream &out) const;
};

class Dirac : public stf {

public:
  KOKKOS_FUNCTION Dirac(type_real f0, type_real tshift, type_real factor,
                        bool use_trick_for_better_pressure);
  KOKKOS_FUNCTION type_real compute(type_real t) override;
  KOKKOS_FUNCTION void update_tshift(type_real tshift) override {
    this->tshift = tshift;
  }
  KOKKOS_FUNCTION type_real get_t0() const override { return this->t0; }
  // void print(std::ostream &out) const override;

private:
  type_real f0;
  type_real tshift;
  type_real t0;
  type_real factor;
  bool use_trick_for_better_pressure;
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::forcing_function::stf &stf);

struct stf_storage {
  specfem::forcing_function::stf *T;
};

} // namespace forcing_function
} // namespace specfem

#endif
