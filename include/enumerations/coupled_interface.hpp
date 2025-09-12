/**
 * @file coupled_interface.hpp
 * @brief Defines interface types and attributes for coupling between different
 * physical media
 *
 * This file provides the fundamental infrastructure for multi-physics coupling
 * in SPECFEM++, enabling simulation of wave propagation across interfaces
 * between different medium types such as elastic-acoustic boundaries. The
 * coupling system supports weakly conforming interfaces where different
 * physical equations are solved on either side of the boundary.
 *
 * @author SPECFEM++ Development Team
 * @date 2025
 * @copyright Princeton University
 */

#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

/**
 * @brief Forward declarations for point field types
 *
 * These forward declarations allow the interface system to reference field
 * types without requiring full inclusion of the field headers, reducing
 * compilation dependencies and enabling circular reference resolution.
 */
// Forward declaration for point
namespace specfem::point {

/**
 * @brief Acceleration field template for various medium types and dimensions
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam MediumTag Physical medium type (elastic, acoustic, etc.)
 * @tparam UseSIMD Whether to use SIMD optimizations
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct acceleration;

/**
 * @brief Displacement field template for various medium types and dimensions
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam MediumTag Physical medium type (elastic, acoustic, etc.)
 * @tparam UseSIMD Whether to use SIMD optimizations
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct displacement;

} // namespace specfem::point

/**
 * @namespace specfem::interface
 * @brief Interface coupling system for multi-physics simulations
 *
 * This namespace contains the core infrastructure for coupling different
 * physical media in SPECFEM++. It provides type-safe mechanisms to define
 * interfaces between different medium types (elastic, acoustic, etc.) and
 * automatically determine the appropriate field types and coupling operations
 * for each interface configuration.
 *
 * The coupling system is designed around the concept of "self" and "coupled"
 * fields, where each interface has a primary medium (self) that receives
 * contributions from a secondary medium (coupled) across the interface
 * boundary.
 */
namespace specfem::interface {

/**
 * @enum interface_tag
 * @brief Enumeration of supported interface coupling types
 *
 * These tags define the direction and type of coupling between different
 * physical media. Each tag specifies which medium is the "self" field
 * (receiving the coupling) and which is the "coupled" field (providing the
 * coupling source).
 *
 * @note The coupling is directional - elastic_acoustic means coupling from
 * elastic to acoustic medium, while acoustic_elastic means the reverse
 * direction.
 */
enum class interface_tag {
  elastic_acoustic, ///< Elastic to acoustic interface - elastic field couples
                    ///< to acoustic
  acoustic_elastic  ///< Acoustic to elastic interface - acoustic field couples
                    ///< to elastic
};

/**
 * @brief Interface attributes template for type-safe field determination
 * @tparam DimensionTag Spatial dimension of the simulation (2D or 3D)
 * @tparam InterfaceTag Type of interface coupling (elastic_acoustic,
 * acoustic_elastic)
 *
 * This template class provides compile-time determination of the appropriate
 * field types and medium tags for a given interface configuration. It ensures
 * type safety and consistency across the coupling system by automatically
 * selecting the correct field types based on the interface specification.
 */
template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag>
class attributes;

/**
 * @brief Attributes specialization for 2D elastic-to-acoustic coupling
 *
 * This specialization defines the field types and medium tags for coupling from
 * elastic media to acoustic media in 2D simulations. In this configuration:
 * - Self medium: elastic_psv (receives coupling contributions)
 * - Coupled medium: acoustic (provides coupling source)
 * - Self field: elastic acceleration (vector field)
 * - Coupled field: acoustic acceleration (scalar field)
 *
 * The coupling typically involves projecting acoustic pressure accelerations
 * onto the elastic medium through the interface normal vector.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::interface::interface_tag::elastic_acoustic> {
public:
  /**
   * @brief Get the medium tag for the self field (receiving coupling)
   * @return Medium tag for elastic PSV medium
   */
  static constexpr specfem::element::medium_tag self_medium() {
    return specfem::element::medium_tag::elastic_psv;
  }

  /**
   * @brief Get the medium tag for the coupled field (providing coupling)
   * @return Medium tag for acoustic medium
   */
  static constexpr specfem::element::medium_tag coupled_medium() {
    return specfem::element::medium_tag::acoustic;
  }

  /**
   * @brief Self field type templates for different connection types
   * @tparam ConnectionTag Type of mesh connectivity (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct self_field;

  /**
   * @brief Coupled field type templates for different connection types
   * @tparam ConnectionTag Type of mesh connectivity (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct coupled_field;

  /**
   * @brief Type alias for self field based on connection type
   * @tparam ConnectionTag Type of mesh connectivity
   */
  template <specfem::connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /**
   * @brief Type alias for coupled field based on connection type
   * @tparam ConnectionTag Type of mesh connectivity
   */
  template <specfem::connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};

template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::elastic_acoustic>::
    self_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   false>;
};

template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::elastic_acoustic>::
    coupled_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   false>;
};

/**
 * @brief Attributes specialization for 2D acoustic-to-elastic coupling
 *
 * This specialization defines the field types and medium tags for coupling from
 * acoustic media to elastic media in 2D simulations. In this configuration:
 * - Self medium: acoustic (receives coupling contributions)
 * - Coupled medium: elastic_psv (provides coupling source)
 * - Self field: acoustic acceleration (scalar field)
 * - Coupled field: elastic displacement (vector field)
 *
 * The coupling typically involves projecting elastic displacement vectors
 * onto the acoustic medium through the interface normal vector, converting
 * vector quantities to scalar pressure contributions.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::interface::interface_tag::acoustic_elastic> {
public:
  /**
   * @brief Get the medium tag for the self field (receiving coupling)
   * @return Medium tag for acoustic medium
   */
  static constexpr specfem::element::medium_tag self_medium() {
    return specfem::element::medium_tag::acoustic;
  }

  /**
   * @brief Get the medium tag for the coupled field (providing coupling)
   * @return Medium tag for elastic PSV medium
   */
  static constexpr specfem::element::medium_tag coupled_medium() {
    return specfem::element::medium_tag::elastic_psv;
  }

  /**
   * @brief Self field type templates for different connection types
   * @tparam ConnectionTag Type of mesh connectivity (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct self_field;

  /**
   * @brief Coupled field type templates for different connection types
   * @tparam ConnectionTag Type of mesh connectivity (weakly_conforming, etc.)
   */
  template <specfem::connections::type ConnectionTag> struct coupled_field;

  /**
   * @brief Type alias for self field based on connection type
   * @tparam ConnectionTag Type of mesh connectivity
   */
  template <specfem::connections::type ConnectionTag>
  using self_field_t = typename self_field<ConnectionTag>::type;

  /**
   * @brief Type alias for coupled field based on connection type
   * @tparam ConnectionTag Type of mesh connectivity
   */
  template <specfem::connections::type ConnectionTag>
  using coupled_field_t = typename coupled_field<ConnectionTag>::type;
};

template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::acoustic_elastic>::
    self_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   false>;
};

template <>
struct attributes<specfem::dimension::type::dim2,
                  specfem::interface::interface_tag::acoustic_elastic>::
    coupled_field<specfem::connections::type::weakly_conforming> {
  using type =
      specfem::point::displacement<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   false>;
};

} // namespace specfem::interface
