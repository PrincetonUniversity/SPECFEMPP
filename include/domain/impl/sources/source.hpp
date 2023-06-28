#ifndef DOMAIN_SOURCE_ELEMENTS_HPP
#define DOMAIN_SOURCE_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Source
 *
 * @tparam properties Properties of the source
 */
template <class... properties> class source {};

} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
