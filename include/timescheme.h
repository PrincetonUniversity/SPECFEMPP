#ifndef TIMESCHEME_H
#define TIMESCHEME_H

#include "../include/config.h"
// #include "../include/domain.h"
#include <ostream>

namespace specfem {
namespace TimeScheme {
class TimeScheme {

public:
  virtual bool status() const { return false; };
  virtual void update_time(){};
  virtual type_real get_time() const { return 0.0; }
  virtual int get_timestep() const { return 0; }
  virtual void reset_time(){};
  // virtual void update_fields(specfem::Domain::Domain *domain_class){};
  virtual int get_max_time() { return 0; }
  // virtual void apply_predictor_phase(specfem::Domain::Domain *domain_class)
  // {}; virtual void apply_corrector_phase(specfem::Domain::Domain
  // *domain_class) {};
  friend std::ostream &operator<<(std::ostream &out, TimeScheme &ts);
  virtual void print(std::ostream &out) const;
};

class Newmark : public TimeScheme {

public:
  Newmark(int nstep, type_real t0, type_real dt);
  bool status() const override { return (this->istep < this->nstep); }
  void update_time() override;
  type_real get_time() const override { return this->current_time; }
  int get_timestep() const override { return this->istep; }
  void reset_time() override;
  // void update_fields(specfem::Domain::Domain *domain_class){};
  int get_max_time() override { return this->nstep; }
  // void apply_predictor_phase(specfem::Domain::Domain *domain_class) override;
  // void apply_corrector_phase(specfem::Domain::Domain *domain_class) override;

  void print(std::ostream &out) const override;

private:
  type_real current_time;
  int istep = 0;
  type_real deltat;
  type_real deltatover2;
  type_real deltatsquareover2;
  int nstep;
  type_real t0;
};

std::ostream &operator<<(std::ostream &out,
                         specfem::TimeScheme::TimeScheme &ts);
std::ostream &operator<<(std::ostream &out,
                         specfem::TimeScheme::Newmark &newmark);

} // namespace TimeScheme
} // namespace specfem
#endif
