#ifndef DOMAIN_SOURCE_ELEMENTS_HPP
#define DOMAIN_SOURCE_ELEMENTS_HPP

#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemental source class
 *
 * Elemental source class to describe the source contribution to the global
 * force vector.
 *
 * @tparam properties Properties of the source
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          typename quadrature_points_type>
class source;

} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
