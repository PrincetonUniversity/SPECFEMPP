#ifndef _TIME_MARCHING_HPP
#define _TIME_MARCHING_HPP

#include "coupled_interface/interface.hpp"
#include "domain/interface.hpp"
#include "enumerations/interface.hpp"
#include "solver.hpp"
#include "timescheme/interface.hpp"

namespace specfem {
namespace solver {
// template <typename Kernels, typename TimeScheme>
// class time_marching : public solver, public Kernels, public TimeScheme {
// public:
//   time_marching(const Kernels &kernels, const TimeScheme &time_scheme)
//       : Kernels(kernels), TimeScheme(time_scheme) {}

//   void run() override;
// };
// } // namespace solver
// } // namespace specfem

/**
 * @brief Time marching solver class
 *
 * Implements a forward time marching scheme given a time scheme and domains.
 * Currently only acoustic and elastic domains are supported.
 *
 * @tparam qp_type Type defining number of quadrature points either at
 compile
 * time or run time
 */
template <typename qp_type>
class time_marching : public specfem::solver::solver {

public:
  constexpr static auto Dimension = specfem::dimension::type::dim2;
  constexpr static auto ElasticTag = specfem::element::medium_tag::elastic;
  constexpr static auto AcousticTag = specfem::element::medium_tag::acoustic;
  constexpr static auto IsotropicTag =
      specfem::element::property_tag::isotropic;
  using elastic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic>;
  using acoustic_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::acoustic>;
  constexpr static auto forward_type = specfem::simulation::type::forward;
  /**
   * @brief Construct a new time marching solver object
   *
   * @param acoustic_domain domain object template specialized for acoustic
   * media
   * @param elastic_domain domain object template specialized for elastic
   media
   * @param it Pointer to time scheme object (it stands for iterator)
   */
  time_marching(
      const specfem::compute::assembly &assembly,
      const specfem::domain::domain<forward_type, Dimension, AcousticTag,
                                    qp_type> &acoustic_domain,
      const specfem::domain::domain<forward_type, Dimension, ElasticTag,
                                    qp_type> &elastic_domain,
      const specfem::coupled_interface::coupled_interface<
          Dimension, AcousticTag, ElasticTag> &acoustic_elastic_interface,
      const specfem::coupled_interface::coupled_interface<
          Dimension, ElasticTag, AcousticTag> &elastic_acoustic_interface,
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
  specfem::compute::simulation_field<forward_type> forward_field;
  specfem::domain::domain<forward_type, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic, qp_type>
      acoustic_domain;
  specfem::domain::domain<forward_type, specfem::dimension::type::dim2,
                          specfem::element::medium_tag::elastic, qp_type>
      elastic_domain;
  specfem::coupled_interface::coupled_interface<
      specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
      specfem::element::medium_tag::elastic>
      acoustic_elastic_interface; ///< Acoustic-elastic interface object
  specfem::coupled_interface::coupled_interface<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
      specfem::element::medium_tag::acoustic>
      elastic_acoustic_interface; ///< Elastic-acoustic interface object
  std::shared_ptr<specfem::TimeScheme::TimeScheme>
      it; ///< Pointer to
          ///< spectem::TimeScheme::TimeScheme
          ///< class
};
} // namespace solver
} // namespace specfem

#endif
