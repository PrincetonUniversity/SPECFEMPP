#ifndef SOLVER_H
#define SOLVER_H

#include "../include/domain.h"
#include "../include/timescheme.h"

namespace specfem {
namespace solver {

/**
 * @brief Base solver class
 *
 */
class solver {

public:
  /**
   * @brief Run solver algorithm
   *
   */
  virtual void run(){};
};

class time_marching : public solver {

public:
  /**
   * @brief Construct a new time marching solver object
   *
   * @param domain Pointer to specfem::Domain::Domain class
   * @param it Pointer to spectem::TimeScheme::TimeScheme class
   */
  time_marching(specfem::Domain::Domain *domain,
                specfem::TimeScheme::TimeScheme *it)
      : domain(domain), it(it){};
  /**
   * @brief Run time-marching solver algorithm
   *
   */
  void run() override;

private:
  specfem::Domain::Domain *domain; ///< Pointer to spefem::Domain::Domain class
  specfem::TimeScheme::TimeScheme *it; ///< Pointer to
                                       ///< spectem::TimeScheme::TimeScheme
                                       ///< class
};
} // namespace solver
} // namespace specfem

#endif
