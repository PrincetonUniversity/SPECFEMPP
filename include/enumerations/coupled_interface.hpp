#pragma once

#include "enumerations/interface.hpp"

namespace specfem::point {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct acceleration;

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct displacement;

} // namespace specfem::point

namespace specfem::interface {

/**
 * @brief Interface coupling direction types.
 *
 * Directional coupling: elastic_acoustic (elastic→acoustic),
 * acoustic_elastic (acoustic→elastic).
 */
enum class interface_tag {
  elastic_acoustic, ///< Elastic to acoustic interface - elastic field couples
                    ///< to acoustic
  acoustic_elastic  ///< Acoustic to elastic interface - acoustic field couples
                    ///< to elastic
};

/**
 * @brief Compile-time interface field type determination.
 *
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam InterfaceTag Interface coupling type
 *
 * @code
 * using attrs = attributes<dim2, interface_tag::elastic_acoustic>;
 * static_assert(attrs::self_medium() == medium_tag::elastic_psv);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
class attributes;

/**
 * @brief 2D elastic→acoustic coupling attributes.
 *
 * Self: elastic_psv (vector acceleration), Coupled: acoustic (scalar
 * acceleration).
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::interface::interface_tag::elastic_acoustic> {
public:
  /**
   * @brief Self medium (receives coupling).
   * @return elastic_psv medium tag
   */
  static constexpr specfem::element::medium_tag self_medium() {
    return specfem::element::medium_tag::elastic_psv;
  }

  /**
   * @brief Coupled medium (provides coupling).
   * @return acoustic medium tag
   */
  static constexpr specfem::element::medium_tag coupled_medium() {
    return specfem::element::medium_tag::acoustic;
  }

  /**
   * @brief Self field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct self_field;

  /**
   * @brief Coupled field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct coupled_field;

  /// Type alias for self field
  template <specfem::connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /// Type alias for coupled field
  template <specfem::connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};

/**
 * @brief 2D weakly conforming elastic→acoustic self field type.
 */
template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::elastic_acoustic>::
    self_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   false>; ///< vector acceleration
};

/**
 * @brief 2D weakly conforming elastic→acoustic coupled field type.
 */
template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::elastic_acoustic>::
    coupled_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   false>; ///< scalar acceleration
};

/**
 * @brief 2D acoustic→elastic coupling attributes.
 *
 * Self: acoustic (scalar acceleration), Coupled: elastic_psv (vector
 * displacement).
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::interface::interface_tag::acoustic_elastic> {
public:
  /**
   * @brief Self medium (receives coupling).
   * @return acoustic medium tag
   */
  static constexpr specfem::element::medium_tag self_medium() {
    return specfem::element::medium_tag::acoustic;
  }

  /**
   * @brief Coupled medium (provides coupling).
   * @return elastic_psv medium tag
   */
  static constexpr specfem::element::medium_tag coupled_medium() {
    return specfem::element::medium_tag::elastic_psv;
  }

  /**
   * @brief Self field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct self_field;

  /**
   * @brief Coupled field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct coupled_field;

  /// Type alias for self field
  template <specfem::connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /// Type alias for coupled field
  template <specfem::connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};

/**
 * @brief 2D weakly conforming acoustic→elastic self field type.
 */
template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::acoustic_elastic>::
    self_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   false>; ///< scalar acceleration
};

/**
 * @brief 2D weakly conforming acoustic→elastic coupled field type.
 */
template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::acoustic_elastic>::
    coupled_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::displacement<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   false>; ///< vector displacement
};

/**
 * @brief Convert interface tag to string.
 * @param interface_tag Interface coupling type
 * @return String representation ("elastic_acoustic" or "acoustic_elastic")
 */
std::string to_string(const interface_tag &interface_tag);

} // namespace specfem::interface
