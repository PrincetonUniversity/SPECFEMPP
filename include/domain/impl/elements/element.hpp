#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Element
 *
 * @tparam properties Properties of the element
 */
template <class... properties> class element {};

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
