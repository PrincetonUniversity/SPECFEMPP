#pragma once

#include "specfem/element.hpp"

namespace specfem::element_coupling {
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
template <specfem::element::dimension_tag DimensionTag,
          specfem::element_coupling::interface_tag InterfaceTag>
class attributes;

/**
 * @brief 2D elastic→acoustic coupling attributes.
 *
 * Self: elastic_psv (vector acceleration), Coupled: acoustic (scalar
 * acceleration).
 */
template <specfem::element::dimension_tag DimensionTag>
class attributes<DimensionTag,
                 specfem::element_coupling::interface_tag::elastic_acoustic> {
public:
  /**
   * @brief Self medium (receives coupling).
   * @return elastic_psv medium tag
   */
  static constexpr specfem::element::medium_tag self_medium() {
    if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
      return specfem::element::medium_tag::elastic_psv;
    } else {
      return specfem::element::medium_tag::elastic;
    }
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
  template <specfem::element_connections::type ConnectionTag> struct self_field;

  /**
   * @brief Weakly conforming elastic→acoustic self field type.
   */
  template <>
  struct self_field<specfem::element_connections::type::weakly_conforming> {
    using type = std::conditional_t<
        DimensionTag == specfem::element::dimension_tag::dim2,
        specfem::point::acceleration<DimensionTag,
                                     specfem::element::medium_tag::elastic_psv,
                                     false>, ///< vector acceleration
        specfem::point::acceleration<DimensionTag,
                                     specfem::element::medium_tag::elastic,
                                     false> ///< vector acceleration
        >;
  };

  /**
   * @brief Coupled field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::element_connections::type ConnectionTag>
  struct coupled_field;

  /**
   * @brief Weakly conforming elastic→acoustic coupled field type.
   */
  template <>
  struct coupled_field<specfem::element_connections::type::weakly_conforming> {
    using type =
        specfem::point::acceleration<DimensionTag,
                                     specfem::element::medium_tag::acoustic,
                                     false>; ///< scalar acceleration
  };

  /// Type alias for self field
  template <specfem::element_connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /// Type alias for coupled field
  template <specfem::element_connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};

/**
 * @brief 2D acoustic→elastic coupling attributes.
 *
 * Self: acoustic (scalar acceleration), Coupled: elastic_psv (vector
 * displacement).
 */
template <specfem::element::dimension_tag DimensionTag>
class attributes<DimensionTag,
                 specfem::element_coupling::interface_tag::acoustic_elastic> {
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
    if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
      return specfem::element::medium_tag::elastic_psv;
    } else {
      return specfem::element::medium_tag::elastic;
    }
  }

  /**
   * @brief Self field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::element_connections::type ConnectionTag> struct self_field;

  /**
   * @brief Weakly conforming acoustic→elastic self field type.
   */
  template <>
  struct self_field<specfem::element_connections::type::weakly_conforming> {
    using type =
        specfem::point::acceleration<DimensionTag,
                                     specfem::element::medium_tag::acoustic,
                                     false>; ///< scalar acceleration
  };

  /**
   * @brief Coupled field type for connection types.
   * @tparam ConnectionTag Connection type (weakly_conforming, etc.)
   */
  template <specfem::element_connections::type ConnectionTag>
  struct coupled_field;

  /**
   * @brief Weakly conforming acoustic→elastic coupled field type.
   */
  template <>
  struct coupled_field<specfem::element_connections::type::weakly_conforming> {
    using type = std::conditional_t<
        DimensionTag == specfem::element::dimension_tag::dim2,
        specfem::point::displacement<DimensionTag,
                                     specfem::element::medium_tag::elastic_psv,
                                     false>, ///< vector displacement
        specfem::point::displacement<DimensionTag,
                                     specfem::element::medium_tag::elastic,
                                     false> ///< vector displacement
        >;
  };

  /// Type alias for self field
  template <specfem::element_connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /// Type alias for coupled field
  template <specfem::element_connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};
} // namespace specfem::element_coupling
