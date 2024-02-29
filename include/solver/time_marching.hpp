#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "coupled_interface/interface.hpp"
#include "domain/interface.hpp"
#include "enumerations/interface.hpp"
#include "solver.hpp"
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
  using elastic_type = specfem::enums::element::medium::elastic;
  using acoustic_type = specfem::enums::element::medium::acoustic;
  constexpr static auto forward_type =
      specfem::enums::simulation::type::forward;
  /**
   * @brief Construct a new time marching solver object
   *
   * @param acoustic_domain domain object template specialized for acoustic
   * media
   * @param elastic_domain domain object template specialized for elastic media
   * @param it Pointer to time scheme object (it stands for iterator)
   */
  time_marching(
      const specfem::compute::assembly &assembly,
      const specfem::domain::domain<acoustic_type, qp_type> &acoustic_domain,
      const specfem::domain::domain<elastic_type, qp_type> &elastic_domain,
      const specfem::coupled_interface::coupled_interface<
          acoustic_type, elastic_type> &acoustic_elastic_interface,
      const specfem::coupled_interface::coupled_interface<
          elastic_type, acoustic_type> &elastic_acoustic_interface,
      std::shared_ptr<specfem::TimeScheme::TimeScheme> it)
      : forward_field(assembly.fields.forward),
        acoustic_domain(acoustic_domain), elastic_domain(elastic_domain),
        acoustic_elastic_interface(acoustic_elastic_interface),
        elastic_acoustic_interface(elastic_acoustic_interface), it(it){};
  /**
   * @brief Run time-marching solver algorithm
   *
   */
  void run() override;

private:
  specfem::compute::simulation_field<forward_type> forward_field;  ///< Fields
                                                                   ///< object
  specfem::domain::domain<acoustic_type, qp_type> acoustic_domain; ///< Acoustic
                                                                   ///< domain
                                                                   ///< object
  specfem::domain::domain<elastic_type, qp_type> elastic_domain;   ///< Elastic
                                                                   ///< domain
                                                                   ///< object
  specfem::coupled_interface::coupled_interface<acoustic_type, elastic_type>
      acoustic_elastic_interface; ///< Acoustic-elastic interface object
  specfem::coupled_interface::coupled_interface<elastic_type, acoustic_type>
      elastic_acoustic_interface; ///< Elastic-acoustic interface object
  std::shared_ptr<specfem::TimeScheme::TimeScheme>
      it; ///< Pointer to
          ///< spectem::TimeScheme::TimeScheme
          ///< class
};
} // namespace solver
} // namespace specfem

#endif
