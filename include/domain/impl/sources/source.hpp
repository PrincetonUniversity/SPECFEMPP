#ifndef DOMAIN_SOURCE_ELEMENTS_HPP
#define DOMAIN_SOURCE_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemental source class
 *
 * Elemental source class contains methods used to compute the elemental source
 * contribution to the global force vector. The class computes how the source
 * interacts with a particular type of element.
 *
 * Elemental sources are implementated as template specializations. Having
 * specialized elemental sources allows for a more flexible and efficient
 * implementation of the methods.
 *
 * @tparam properties Properties of the source
 */
template <class... properties> class source {
  using dimension = void;
  using medium_type = void;
  using quadrature_points_type = void;
};

} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
