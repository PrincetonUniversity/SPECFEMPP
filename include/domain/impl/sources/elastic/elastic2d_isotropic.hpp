#ifndef _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP
#define _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP

#include "compute/interface.hpp"
// #include "domain/impl/sources/elastic/elastic2d.hpp"
#include "domain/impl/sources/source.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemental source specialization for 2D elastic isotropic spectral
 * elements with static quadrature points
 *
 * @tparam NGLL Number of Gauss-Lobatto-Legendre quadrature points defined at
 * compile time
 */
template <int NGLL>
class source<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::property_tag::isotropic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL> > {

public:
  /**
   * @name Typedefs
   */
  ///@{
  using dimension =
      specfem::dimension::dimension<specfem::dimension::type::dim2>;
  using medium_type =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::isotropic>;

  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points
   */
  using quadrature_points_type =
      specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
  ///@}

  /**
   * @brief Default elemental source constructor
   *
   */
  KOKKOS_FUNCTION source() = default;

  /**
   * @brief Default elemental source copy constructor
   *
   */
  KOKKOS_FUNCTION source(const source &) = default;

  //   /**
  //    * @brief Construct a new elemental source object
  //    *
  //    * @param source_array Source array containing pre-computed lagrange
  //    * interpolants
  //    */
  //   KOKKOS_FUNCTION source(const specfem::compute::properties &properties,
  //                          specfem::kokkos::DeviceView4d<type_real>
  //                          source_array);

  /**
   * @brief Compute the interaction of the source with the medium computed at
   * the quadrature point xz
   *
   * @param isource Index of the source
   * @param ispec Index of the element
   * @param xz Quadrature point index in the element
   * @param stf_value Value of the source time function at the current time step
   * @param acceleration Acceleration contribution to the global force vector by
   * the source
   */
  KOKKOS_INLINE_FUNCTION void compute_interaction(
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &stf,
      const specfem::kokkos::array_type<type_real, medium_type::components>
          &lagrange_interpolant,
      specfem::kokkos::array_type<type_real, medium_type::components>
          &acceleration) const;
};
} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
