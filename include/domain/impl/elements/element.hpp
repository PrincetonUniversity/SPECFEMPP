#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Element class to describe the physics of a spectral element
 *
 * @tparam properties of the element used to specialize elemental implementation
 */
template <class... properties> class element {};

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
