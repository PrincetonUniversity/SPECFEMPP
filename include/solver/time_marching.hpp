#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "domain/interface.hpp"
#include "solver.hpp"
#include "specfem_enums.hpp"
#include "timescheme/interface.hpp"

namespace specfem {
namespace solver {
/**
 * @brief Time marching solver class
 *
 * Implements a forward time marching scheme given a time scheme and domains.
 * Currently only acoustic and elastic domains are supported.
 *
 * @tparam qp_type Type defining number of quadrature points either at compile
 * time or run time
 */
template <typename qp_type>
class time_marching : public specfem::solver::solver {

public:
  /**
   * @brief Construct a new time marching solver object
   *
   * @param acoustic_domain domain object template specialized for acoustic
   * media
   * @param elastic_domain domain object template specialized for elastic media
   * @param it Pointer to time scheme object (it stands for iterator)
   */
  time_marching(
      specfem::domain::domain<specfem::enums::element::medium::acoustic,
                              qp_type>
          acoustic_domain,
      specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
          elastic_domain,
      specfem::TimeScheme::TimeScheme *it)
      : acoustic_domain(acoustic_domain), elastic_domain(elastic_domain),
        it(it){};
  /**
   * @brief Run time-marching solver algorithm
   *
   */
  void run() override;

private:
  specfem::domain::domain<specfem::enums::element::medium::acoustic, qp_type>
      acoustic_domain; ///< Pointer to spefem::Domain::Domain class
  specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
      elastic_domain; ///< Pointer to spefem::Domain::Domain class
  specfem::TimeScheme::TimeScheme *it; ///< Pointer to
                                       ///< spectem::TimeScheme::TimeScheme
                                       ///< class
};
} // namespace solver
} // namespace specfem

#endif
