#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "domain.h"
#include "solver.hpp"
#include "timescheme.h"

namespace specfem {
namespace solver {
class time_marching : public specfem::solver::solver {

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
