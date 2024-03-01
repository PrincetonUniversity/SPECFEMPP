#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property,
    specfem::element::boundary_tag boundary, typename quadrature_points_type>
class element;

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
