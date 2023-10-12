#ifndef _ENUMERATIONS_PROPERTIES_HPP_
#define _ENUMERATIONS_PROPERTIES_HPP_

namespace specfem {
namespace enums {
namespace element {
/**
 * @namespace Elemental properties
 *
 * Properties can be utilized to distinguish elements based on physics or to
 * optimize kernel calculations for specific elements.
 */
namespace property {
/**
 * @brief Isotropic element
 *
 */
class isotropic {};
} // namespace property
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_PROPERTIES_HPP_ */
