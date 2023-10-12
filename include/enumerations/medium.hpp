#ifndef _ENUMERATIONS_MEDIUM_HPP_
#define _ENUMERATIONS_MEDIUM_HPP_

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
      specfem::enums::element::elastic;
  /**
   * @brief Number of components for this medium
   *
   */
  constexpr static int components = 2;
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
      specfem::enums::element::acoustic;
  /**
   * @brief constexpr defining number of components for this medium.
   *
   */
  constexpr static int components = 1;
};

} // namespace medium
} // namespace element
} // namespace enums
} // namespace specfem

#endif /* _ENUMERATIONS_MEDIUM_HPP_ */
