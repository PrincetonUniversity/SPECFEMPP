#ifndef DOMAIN_ELEMENTS_HPP
#define DOMAIN_ELEMENTS_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Element class
 *
 * Elements are the building blocks of the compute infratructure within a
 * domain. Each instance of an element class respresent a single spectral
 * element within the domain. The class itself provides methods to compute the
 * element's contribution to the global force vector. The methods included in
 * this class @code compute_gradients, compute_stresses, update_acceleration
 * @endcode compute the relevant quantities at the quadrature points of the
 * element.
 *
 * Elements are implementated as template specializations. The template
 * parameters are used to specify the dimensionality of the element, the medium,
 * the quadrature points and the properties of the element. Having specialized
 * elements allows for a more flexible and efficient implementation of the
 * methods.
 *
 * @tparam properties Properties of the element
 */
template <class... properties> class element {};

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
