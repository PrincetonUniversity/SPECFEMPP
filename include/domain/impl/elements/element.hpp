#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
class element;

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
