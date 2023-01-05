#ifndef SOLVER_H
#define SOLVER_H

#include "../include/domain.h"
#include "../include/timescheme.h"

namespace specfem {
namespace solver {
class solver {
public:
  virtual void run(){};
};

class time_marching : public solver {

public:
  time_marching(specfem::Domain::Domain *domain,
                specfem::TimeScheme::TimeScheme *it)
      : domain(domain), it(it){};
  void run() override;

private:
  specfem::Domain::Domain *domain;
  specfem::TimeScheme::TimeScheme *it;
};
} // namespace solver
} // namespace specfem

#endif
