#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "domain/interface.hpp"
#include "solver.hpp"
#include "specfem_enums.hpp"
#include "timescheme/interface.hpp"

namespace specfem {
namespace solver {
template <typename qp_type>
class time_marching : public specfem::solver::solver {

public:
  /**
   * @brief Construct a new time marching solver object
   *
   * @param domain Pointer to specfem::Domain::Domain class
   * @param it Pointer to spectem::TimeScheme::TimeScheme class
   */
  time_marching(
      specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
          elastic_domain,
      specfem::TimeScheme::TimeScheme *it)
      : elastic_domain(elastic_domain), it(it){};
  /**
   * @brief Run time-marching solver algorithm
   *
   */
  void run() override;

private:
  specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
      elastic_domain; ///< Pointer to spefem::Domain::Domain class
  specfem::TimeScheme::TimeScheme *it; ///< Pointer to
                                       ///< spectem::TimeScheme::TimeScheme
                                       ///< class
};
} // namespace solver
} // namespace specfem

#endif
