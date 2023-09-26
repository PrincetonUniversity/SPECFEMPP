#ifndef _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP
#define _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP

#include "compute/interface.hpp"
#include "domain/impl/sources/elastic/elastic2d.hpp"
#include "domain/impl/sources/source.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

/**
 * @brief Decltype for the field subviewed at particular global index
 *
 */
using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemenatal source class for 2D isotropic elastic medium with number of
 * quadrature points defined at compile time
 *
 * @tparam N Number of Gauss-Lobatto-Legendre quadrature points
 */
template <int N>
class source<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::elastic,
             specfem::enums::element::quadrature::static_quadrature_points<N>,
             specfem::enums::element::property::isotropic> {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = specfem::enums::element::medium::elastic;
  using quadrature_points_type =
      specfem::enums::element::quadrature::static_quadrature_points<N>;
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

  /**
   * @brief Construct a new elemental source object
   *
   * @param ispec Index of the element where the source is located
   * @param source_array Source array containing pre-computed lagrange
   * interpolants
   * @param stf Pointer to the source time function object
   */
  KOKKOS_FUNCTION source(const specfem::compute::properties &properties,
                         specfem::kokkos::DeviceView4d<type_real> source_array);

  /**
   * @brief Compute the interaction of the source with the medium computed at
   * the quadrature point xz
   *
   * @param xz Quadrature point index in the element
   * @param stf_value Value of the source time function at the current time step
   * @param acceleration Acceleration contribution to the global force vector by
   * the source
   */
  KOKKOS_INLINE_FUNCTION void
  compute_interaction(const int &isource, const int &ispec, const int &xz,
                      const type_real &stf_value,
                      type_real *acceleration) const;

private:
  specfem::kokkos::DeviceView4d<type_real> source_array; ///< Source array
                                                         ///< containing
                                                         ///< pre-computed
                                                         ///< lagrange
                                                         ///< interpolants
};
} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
