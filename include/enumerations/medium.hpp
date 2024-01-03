#ifndef _ENUMERATIONS_MEDIUM_HPP_
#define _ENUMERATIONS_MEDIUM_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace enums {
namespace element {

/**
 * @namespace medium property of the element
 *
 */
namespace medium {
/**
 * @brief Elastic medium
 *
 */
class elastic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::type::elastic;
  /**
   * @brief Number of components for this medium
   *
   */
  constexpr static int components = 2;

  /**
   * @brief Convert the medium to a string
   *
   */
  __inline__ static std::string to_string() { return "Elastic"; }
};

/**
 * @brief Acoustic medium
 *
 */
class acoustic {
public:
  /**
   * @brief constexpr defining the type of the element
   *
   */
  constexpr static specfem::enums::element::type value =
      specfem::enums::element::type::acoustic;
  /**
   * @brief constexpr defining number of components for this medium.
   *
   */
  constexpr static int components = 1;

  /**
   * @brief Convert the medium to a string
   *
   */
  __inline__ static std::string to_string() { return "Acoustic"; }
};

} // namespace medium
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_MEDIUM_HPP_ */
