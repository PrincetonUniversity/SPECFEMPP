#ifndef DOMAIN_RECEIVER_ELEMENTS_HPP
#define DOMAIN_RECEIVER_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {
/**
 * @brief Elemental source class
 *
 * Elemental source class contains methods used to compute the elemental source
 * contribution to the global force vector.
 *
 * Elemental sources are implementated as template specializations. Having
 * specialized elemental sources allows for a more flexible and efficient
 * implementation of the methods.
 *
 * @tparam properties Properties of the source
 */
template <class... properties> class receiver {

  using dimension = void;
  using medium = void;
  using quadrature_points = void;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
