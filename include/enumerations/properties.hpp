#ifndef _ENUMERATIONS_PROPERTIES_HPP_
#define _ENUMERATIONS_PROPERTIES_HPP_

#include "specfem_enums.hpp"

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
class isotropic {
public:
  constexpr static specfem::enums::element::property_tag value =
      specfem::enums::element::property_tag::isotropic;

  /**
   * @brief Convert the property to a string
   *
   */
  __inline__ static std::string to_string() { return "Isotropic"; }
};
} // namespace property
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_PROPERTIES_HPP_ */
