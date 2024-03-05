#ifndef DOMAIN_RECEIVER_ELEMENTS_HPP
#define DOMAIN_RECEIVER_ELEMENTS_HPP

#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {
/**
 * @brief Elemental receiver class
 *
 * Elemental receiver class to compute the seismogram at the receiver location.
 *
 * @tparam properties Properties of the receiver
 */
template <
    specfem::dimension::type dimension, specfem::element::medium_tag medium,
    specfem::element::property_tag property, typename quadrature_points_type>
class receiver;

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
