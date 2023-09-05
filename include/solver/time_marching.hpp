#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "coupled_interface/interface.hpp"
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
                              qp_type> &acoustic_domain,
      specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
          &elastic_domain,
      specfem::coupled_interface::coupled_interface<
          specfem::domain::domain<specfem::enums::element::medium::acoustic,
                                  qp_type>,
          specfem::domain::domain<specfem::enums::element::medium::elastic,
                                  qp_type> > &acoustic_elastic_interface,
      specfem::coupled_interface::coupled_interface<
          specfem::domain::domain<specfem::enums::element::medium::elastic,
                                  qp_type>,
          specfem::domain::domain<specfem::enums::element::medium::acoustic,
                                  qp_type> > &elastic_acoustic_interface,
      specfem::TimeScheme::TimeScheme *it)
      : acoustic_domain(acoustic_domain), elastic_domain(elastic_domain),
        acoustic_elastic_interface(acoustic_elastic_interface),
        elastic_acoustic_interface(elastic_acoustic_interface), it(it){};
  /**
   * @brief Run time-marching solver algorithm
   *
   */
  void run() override;

private:
  specfem::domain::domain<specfem::enums::element::medium::acoustic, qp_type>
      acoustic_domain; ///< Acoustic domain
  specfem::domain::domain<specfem::enums::element::medium::elastic, qp_type>
      elastic_domain; ///< Acoustic domain
  specfem::coupled_interface::coupled_interface<
      specfem::domain::domain<specfem::enums::element::medium::acoustic,
                              qp_type>,
      specfem::domain::domain<specfem::enums::element::medium::elastic,
                              qp_type> >
      acoustic_elastic_interface; /// Acoustic elastic interface
  specfem::coupled_interface::coupled_interface<
      specfem::domain::domain<specfem::enums::element::medium::elastic,
                              qp_type>,
      specfem::domain::domain<specfem::enums::element::medium::acoustic,
                              qp_type> >
      elastic_acoustic_interface;      /// Elastic acoustic interface
  specfem::TimeScheme::TimeScheme *it; ///< Pointer to
                                       ///< spectem::TimeScheme::TimeScheme
                                       ///< class
};
} // namespace solver
} // namespace specfem

#endif
