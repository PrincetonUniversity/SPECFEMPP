#ifndef SOURCE_TIME_FUNCTION_H
#define SOURCE_TIME_FUNCTION_H

#include "config.h"
#include <ostream>

namespace specfem {
namespace forcing_function {

class stf {
public:
  virtual type_real compute(type_real t) { return 0.0; }
  virtual void update_tshift(type_real tshift){};
  virtual type_real get_t0() const { return 0.0; }

  friend std::ostream &operator<<(std::ostream &out, const stf &stf);

  virtual void print(std::ostream &out) const;
};

class Dirac : public stf {

public:
  Dirac(type_real f0, type_real tshift, type_real factor,
        bool use_trick_for_better_pressure);
  type_real compute(type_real t) override;
  void update_tshift(type_real tshift) override { this->tshift = tshift; }
  type_real get_t0() const override { return this->t0; }
  void print(std::ostream &out) const override;

private:
  type_real f0;
  type_real tshift;
  type_real t0;
  type_real factor;
  bool use_trick_for_better_pressure;
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::forcing_function::stf &stf);

} // namespace forcing_function
} // namespace specfem

#endif
